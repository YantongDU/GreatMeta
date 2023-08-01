# -*- coding: utf-8 -*-
"""
@Time    : 2022/9/14
@Author  : Yantong Du
@Email    : duyantong94@hrbeu.edu.cn
"""
from collections import OrderedDict
from copy import deepcopy

import torch.autograd

from recbole.model.layers import MLPLayers, activation_layer
from torch import nn
import torch.nn.functional as F
from torch.nn.init import normal_

from recbole_metarec.MetaRecommender import MetaRecommender
from recbole.utils import FeatureSource, InputType
from recbole_metarec.MetaUtils import EmbeddingTable, GradCollector


class GreatMeta(MetaRecommender):
    """
    This is implement of paper 'Group-wised meta-learning with an adaptive scheduler for user cold-start recommendation'
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(GreatMeta, self).__init__(config, dataset)

        self.source = [FeatureSource.USER, FeatureSource.ITEM]

        self.MLPHiddenSize = config['mlp_hidden_size']
        self.localLr = config['grad_meta_args']['local_lr']
        self.lr = config['grad_meta_args']['lr']
        self.train_batch_size = config['train_batch_size']

        self.use_avg_grad = config['use_avg_grad']
        self.use_film = config['use_film']
        self.use_loss_instead_of_step_size = config['use_loss_instead_of_step_size']

        self.embeddingTable = EmbeddingTable(self.embedding_size, self.dataset, source=self.source)

        if self.use_film:
            self.userAwareAdapter = UserAdaptingModulator(self.embeddingTable.getAllDim(), config)
        else:
            self.userAwareAdapter = UserAdaptingModulator2(self.embeddingTable.getAllDim(), config)

        self.meta_model = nn.ModuleList([self.embeddingTable, self.userAwareAdapter])
        self.keepWeightParams = deepcopy(self.meta_model.state_dict())
        self.metaGradCollector = GradCollector(list(self.meta_model.state_dict().keys()))

        self.embeddingTableSize = len(self.embeddingTable.state_dict())
        self.userAwareAdapterSize = len(self.userAwareAdapter.state_dict())

        self.taskAwareWeightCalculator = TaskAwareWeightScheduler(self.device, config, self.embeddingTableSize, self.userAwareAdapterSize)
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.taskAwareWeightCalCollector = GradCollector(list(self.taskAwareWeightCalculator.state_dict().keys()))

        self.local_update_count = 3
        self.task_scheduler_optimiser = torch.optim.Adam(self.taskAwareWeightCalculator.parameters(), self.lr)

    def calculate_loss(self, taskBatch):
        # temporary update the metamodel θ by task weight scheduler φ       (θ, φ) ->  θ`
        temp_loss, temp_grad, avgBatchTaskGrad, _ = self.update_meta_model(taskBatch)

        if self.use_avg_grad:
            return temp_loss, avgBatchTaskGrad

        # update the φ with temporary metamodel θ`      (θ`, φ) -> φ*
        self.update_task_scheduler(temp_grad, taskBatch)

        # update the metamodel θ by the task scheduler φ*       (θ, φ*) -> θ*
        batchTaskLoss, final_grad, _, recorder = self.update_meta_model(taskBatch)

        return batchTaskLoss, final_grad, recorder.detach().cpu().numpy()

    def update_task_scheduler(self, final_grad, taskBatch):
        self.meta_model.load_state_dict(self.keepWeightParams)
        newParams = OrderedDict()
        for name, params in self.meta_model.state_dict().items():
            newParams[name] = params - self.lr * final_grad[name]
        # 更新task adapter参数
        paramWeightDict = self.weightParamToParamDict(newParams)
        for index, task in enumerate(taskBatch):
            qrt_x, qrt_y = self.taskDesolveEmb(task, set='qrt', paramWeightDict=paramWeightDict)  # inter_num * 72 (feature_num * embedding_size)

            # 2. 预测每一次交互的rating
            # MLP: 72 * 64 * 64 * 1
            qrt_y_predict = self.userAwareAdapter(qrt_x, paramWeightDict=paramWeightDict)
            # spt_y_predict = self.predictor(spt_x, weightParam)  # inter_num * 1
            # 计算当前task的loss
            loss = F.mse_loss(qrt_y, qrt_y_predict)

            self.taskAwareWeightCalculator.zero_grad()
            weight_grad = torch.autograd.grad(loss, self.taskAwareWeightCalculator.parameters(), create_graph=True)
            # 使用metaGradCollector来收集每一个task的梯度，并在最后计算平均数，将taskBatch中的所有task的梯度一起更新
            # todo: 这里是不是可以每一个task更新一次meta_model
            self.taskAwareWeightCalCollector.addGrad(weight_grad)
        self.taskAwareWeightCalCollector.averageGrad(index+1)
        taskWeightGrad = self.taskAwareWeightCalCollector.dumpGrad()

        # -------------- meta update --------------

        self.task_scheduler_optimiser.zero_grad()

        # set gradients of parameters manually
        for name, param in self.taskAwareWeightCalculator.named_parameters():
            param.grad = taskWeightGrad[name]
            param.grad.data.clamp_(-10, 10)

        # the meta-optimiser only operates on the shared parameters, not the context parameters
        self.task_scheduler_optimiser.step()

    def update_meta_model(self, taskBatch):
        batchTaskLoss = torch.tensor(0.0).to(self.device)
        batchTaskFactor = []
        # 计算每一个task在当前batch中自适应权重
        for taskIndex, task in enumerate(taskBatch):
            global_grad, global_loss, task_grad_tuple = self.forward(task)

            self.metaGradCollector.addAndSaveTaskGrad(taskIndex, global_grad)

            # factor.1 task对于元模型梯度的模长 -- 更新方向不变，更新的步长作为一个调整因素
            # todo: 这里直接使用loss而不是梯度的步长
            if self.use_loss_instead_of_step_size:
                factor_1 = global_loss
            else:
                factor_1 = self.cal_length_of_grad(global_grad)

            # factor.2 task对于当前模型的generation gap
            factor_2 = self.cal_similarity_of_grad(task_grad_tuple)

            taskFactor = [task.taskInfo['user_id'], factor_1.item(), factor_2]
            batchTaskFactor.append(taskFactor)

            batchTaskLoss += global_loss
        # factor.3 每一个task与平均梯度之间的相似度
        self.metaGradCollector.averageGrad(self.train_batch_size)
        avgBatchTaskGrad = self.metaGradCollector.dumpGrad()
        for ((taskIdx, taskGrad), taskFactor) in zip(self.metaGradCollector.taskGradDict.items(), batchTaskFactor):
            factor_3 = self.cal_similarity_of_grad((tuple(avgBatchTaskGrad.values()), taskGrad))  # (21,)
            taskFactor.append(factor_3)
        batchTaskWeight, recorder = self.taskAwareWeightCalculator(batchTaskFactor)
        self.metaGradCollector.addGradWithWeigth(batchTaskWeight)
        final_grad = self.metaGradCollector.dumpGrad(clearTaskGrad=True)
        return batchTaskLoss, final_grad, avgBatchTaskGrad, recorder

    def forward(self, task):
        # 1.加载元模型
        self.meta_model.load_state_dict(self.keepWeightParams)
        modelParamNames = self.userAwareAdapter.state_dict().keys()
        fastWeightParams = OrderedDict()
        # 2.进行bi-level更新，计算task grad与其他的weight factor
        #   2.1.通过embedding layer得到spt和qrt中用户交互的表示 与 qrt在当前模型上的梯度
        spt_x, spt_y, qrt_x, qrt_y = self.taskDesolveEmb(task)  # inter_num * 224 (feature_num * embedding_size)
        #   2.2.进行预测
        qrt_y_predict = self.userAwareAdapter(qrt_x)
        qrt_on_meta_model_loss = F.mse_loss(qrt_y, qrt_y_predict)
        qrt_on_meta_model_grad = torch.autograd.grad(qrt_on_meta_model_loss, self.userAwareAdapter.parameters())

        for _ in range(self.local_update_count):
            originWeightParams = list(self.userAwareAdapter.state_dict().values())
            spt_y_predict = self.userAwareAdapter(spt_x)
            local_loss = F.mse_loss(spt_y, spt_y_predict)
            local_grad = torch.autograd.grad(local_loss, self.userAwareAdapter.parameters(), retain_graph=True)
            for idx, name in enumerate(modelParamNames):
                fastWeightParams[name] = originWeightParams[idx] - self.localLr * local_grad[idx]
            #   2.5.将fast weight更新到模型中去
            self.userAwareAdapter.load_state_dict(fastWeightParams)
        #   2.6.计算qrt在fast weight上的loss
        qrt_x, qrt_y = self.taskDesolveEmb(task, set='qrt')
        # qrt_y_predict = self.predictor(qrt_x)
        qrt_y_predict = self.userAwareAdapter(qrt_x)
        global_loss = F.mse_loss(qrt_y, qrt_y_predict)

        self.userAwareAdapter.zero_grad()
        global_grad_predictor = torch.autograd.grad(global_loss, self.userAwareAdapter.parameters(), retain_graph=True)
        self.embeddingTable.zero_grad()
        global_grad_embedding = torch.autograd.grad(global_loss, self.embeddingTable.parameters())

        global_grad = global_grad_embedding + global_grad_predictor
        # 将fast weight换成meta-model
        self.meta_model.load_state_dict(self.keepWeightParams)
        return global_grad, global_loss, (global_grad_predictor, qrt_on_meta_model_grad)

    def cal_length_of_grad(self, grad):
        length = torch.tensor(0.0, device=self.device)
        for each_grad in grad:
            length += torch.norm(each_grad.flatten().unsqueeze(0))
        return length

    def cal_similarity_of_grad(self, task_grad_tuple):
        grad1 = task_grad_tuple[0]
        grad2 = task_grad_tuple[1]
        # 在计算每一个task梯度和batchTask平均梯度的相似度时候，我们只计算userAwareAdapter部分的相似度
        # if len(grad1) > self.userAwareAdapterSize:
        #     grad1 = grad1[-self.userAwareAdapterSize:]
        #     grad2 = grad2[-self.userAwareAdapterSize:]
        similarity = []
        for idx, _ in enumerate(grad1):
            if len(grad1) > self.userAwareAdapterSize and idx < self.embeddingTableSize:
                similarity.append(torch.matmul(grad1[idx].flatten().unsqueeze(0), grad2[idx].flatten().unsqueeze(0).t())[0].item())
            else:
                similarity.append(self.cosine(grad1[idx].flatten().unsqueeze(0), grad2[idx].flatten().unsqueeze(0))[0].item())
        return torch.tensor(similarity, device=self.device)

    def predict(self, spt_x, spt_y, qrt_x):
        self.meta_model.load_state_dict(self.keepWeightParams)
        modelParamNames = self.meta_model.state_dict().keys()

        spt_x_one_hot = spt_x
        spt_y = spt_y.view(-1, 1)
        for _ in range(self.local_update_count):
            originWeightParams = list(self.meta_model.state_dict().values())
            fastWeightParams = OrderedDict()

            spt_x = self.embeddingAllFields(spt_x_one_hot)
            spt_y_predict = self.userAwareAdapter(spt_x)

            local_loss = F.mse_loss(spt_y, spt_y_predict)
            local_grad = torch.autograd.grad(local_loss, self.meta_model.parameters(), retain_graph=True)
            for idx, name in enumerate(modelParamNames):
                fastWeightParams[name] = originWeightParams[idx] - self.localLr * local_grad[idx]
            #   2.5.将fast weight更新到模型中去
            self.meta_model.load_state_dict(fastWeightParams)
        qrt_x = self.embeddingAllFields(qrt_x)
        qrt_y_predict = self.userAwareAdapter(qrt_x)

        return qrt_y_predict

    def embeddingAllFields(self, interaction, paramWeightDict=None):
        batchX = []
        for field in self.embeddingTable.embeddingFields:
            feature = self.embeddingTable.embeddingSingleField(field, interaction[field], paramWeightDict=paramWeightDict)
            batchX.append(feature)
        batchX = torch.cat(batchX, dim=1)
        return batchX

    def taskDesolveEmb(self, task, set=None, paramWeightDict=None):
        if set == 'spt':
            spt_x = self.embeddingAllFields(task.spt, paramWeightDict)
            spt_y = task.spt[self.RATING].view(-1, 1)
            return spt_x, spt_y
        elif set == 'qrt':
            qrt_x = self.embeddingAllFields(task.qrt, paramWeightDict)
            qrt_y = task.qrt[self.RATING].view(-1, 1)
            return qrt_x, qrt_y
        else:
            spt_x = self.embeddingAllFields(task.spt, paramWeightDict)
            spt_y = task.spt[self.RATING].view(-1, 1)
            qrt_x = self.embeddingAllFields(task.qrt, paramWeightDict)
            qrt_y = task.qrt[self.RATING].view(-1, 1)
            return spt_x, spt_y, qrt_x, qrt_y

    def weightParamToParamDict(self, newParams):
        weightParam = OrderedDict()
        for idx, (name, value) in enumerate(newParams.items()):
            if idx < self.embeddingTableSize:
                k = name.split('.')[1]
                weightParam[k] = value
            # elif idx - self.embeddingTableSize < self.userEncoderSize:
            #     k = name.split('.', 1)[1]
            #     weightParam[k] = value
            # elif idx - self.embeddingTableSize - self.userEncoderSize < self.userAwareAdapterSize:
            #     k = name.split('.', 1)[1]
            #     weightParam[k] = value
            elif idx - self.embeddingTableSize < self.userAwareAdapterSize:
                k = name.split('.', 1)[1]
                weightParam[k] = value
        return weightParam

    def taskDesolveUserEmbedding(self,task):
        spt_x=self.embeddingTable.embeddingUerFields(task.spt)
        return spt_x


class UserAdaptingModulator(nn.Module):
    def __init__(self, emdeddingAllDims, config):
        super(UserAdaptingModulator, self).__init__()
        self.all_feature_dim = emdeddingAllDims
        self.encode_layer_dims = config['encoder_layer_dim']

        self.phi_encoder = GreatMetaSequential(
            Linear(self.all_feature_dim, self.all_feature_dim),
            # nn.Tanh()
        )

        self.predictor_layer_1 = Linear(emdeddingAllDims, self.encode_layer_dims[0])

        self.gama_1_encoder = GreatMetaSequential(
            Linear(emdeddingAllDims, self.encode_layer_dims[0], bias=False),
            # nn.Tanh()
        )
        self.beta_1_encoder = GreatMetaSequential(
            Linear(emdeddingAllDims, self.encode_layer_dims[0], bias=False),
            # nn.Tanh()
        )

        self.predictor_layer_2 = Linear(self.encode_layer_dims[0], self.encode_layer_dims[1])

        self.gama_2_encoder = GreatMetaSequential(
            Linear(emdeddingAllDims, self.encode_layer_dims[1], bias=False),
            # nn.Tanh()
        )
        self.beta_2_encoder = GreatMetaSequential(
            Linear(emdeddingAllDims, self.encode_layer_dims[1], bias=False),
            # nn.Tanh()
        )

        self.predictor_layer_3 = Linear(self.encode_layer_dims[1], 1)

    def forward(self, input, paramWeightDict=None):
        '''
        Args:
            taskEmb: 用户交互embedding，inter_num * 72 (feature_num * embedding_size)
        '''
        taskEmb = self.phi_encoder(torch.mean(input, dim=0), paramWeightDict=paramWeightDict, paramNamePrefix='phi_encoder' if paramWeightDict is not None else None)
        input = self.gama_1_encoder(taskEmb, paramWeightDict=paramWeightDict, paramNamePrefix='gama_1_encoder' if paramWeightDict is not None else None) * self.predictor_layer_1(input, paramWeightDict=paramWeightDict, paramNamePrefix='predictor_layer_1' if paramWeightDict is not None else None) + self.beta_1_encoder(taskEmb, paramWeightDict=paramWeightDict, paramNamePrefix='beta_1_encoder' if paramWeightDict is not None else None)
        input = self.gama_2_encoder(taskEmb, paramWeightDict=paramWeightDict, paramNamePrefix='gama_2_encoder' if paramWeightDict is not None else None) * self.predictor_layer_2(input, paramWeightDict=paramWeightDict, paramNamePrefix='predictor_layer_2' if paramWeightDict is not None else None) + self.beta_2_encoder(taskEmb, paramWeightDict=paramWeightDict, paramNamePrefix='beta_2_encoder' if paramWeightDict is not None else None)
        output = self.predictor_layer_3(input, paramWeightDict=paramWeightDict, paramNamePrefix='predictor_layer_3' if paramWeightDict is not None else None)
        return output


class UserAdaptingModulator2(nn.Module):
    def __init__(self, emdeddingAllDims, config):
        super(UserAdaptingModulator2, self).__init__()
        self.all_feature_dim = emdeddingAllDims
        self.encode_layer_dims = config['encoder_layer_dim']

        self.predictor_layer_1 = Linear(emdeddingAllDims, self.encode_layer_dims[0])

        self.predictor_layer_2 = Linear(self.encode_layer_dims[0], self.encode_layer_dims[1])

        self.predictor_layer_3 = Linear(self.encode_layer_dims[1], 1)

    def forward(self, input, paramWeightDict=None):
        '''
        Args:
            taskEmb: 用户交互embedding，inter_num * 72 (feature_num * embedding_size)
        '''
        # taskEmb = self.phi_encoder(torch.mean(input, dim=0), paramWeightDict=paramWeightDict, paramNamePrefix='phi_encoder' if paramWeightDict is not None else None)
        input = self.predictor_layer_1(input, paramWeightDict=paramWeightDict, paramNamePrefix=self.getPrefix('predictor_layer_1', paramWeightDict))
        input = self.predictor_layer_2(input, paramWeightDict=paramWeightDict, paramNamePrefix=self.getPrefix('predictor_layer_2', paramWeightDict))
        output = self.predictor_layer_3(input, paramWeightDict=paramWeightDict, paramNamePrefix=self.getPrefix('predictor_layer_3', paramWeightDict))
        return output

    def getPrefix(self, modelName, paramDict=None):
        return modelName if paramDict is not None else None

class GreatMetaSequential(nn.Sequential):

    def __init__(self, *args):
        super(GreatMetaSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input, paramWeightDict=None, paramNamePrefix=None):
        sequentialIdx=0
        for module in self:
            if isinstance(module, (Linear, GreatMetaMLPLayers)):
                input = module.forward(input,
                                       paramWeightDict=paramWeightDict,
                                       paramNamePrefix=str(sequentialIdx) if paramNamePrefix is None
                                       else paramNamePrefix + '.' + str(sequentialIdx))
            else:
                input = module(input)
            sequentialIdx += 1
        return input


class GreatMetaMLPLayers(nn.Module):

    def __init__(self, layers, dropout=0., activation='relu', bn=False, init_method=None):
        super(GreatMetaMLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        linear_func = Linear

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(linear_func(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = GreatMetaSequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature, paramWeightDict=None, paramNamePrefix=None):
        return self.mlp_layers(input_feature, paramWeightDict=paramWeightDict, paramNamePrefix=(str(paramNamePrefix) + '.mlp_layers'))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x, paramWeightDict=None, paramNamePrefix=None):
        if paramWeightDict is not None:
            if self.bias is not None:
                out = torch.nn.functional.linear(x, paramWeightDict[paramNamePrefix + '.weight'], paramWeightDict[paramNamePrefix + '.bias']) #weight.fast (fast weight) is the temporaily adapted weight
            else:
                out = torch.nn.functional.linear(x, paramWeightDict[paramNamePrefix + '.weight'])
        else:
            out = super(Linear, self).forward(x)
        return out


class TaskAwareWeightScheduler(nn.Module):
    def __init__(self, device, config, embeddingTableSize, userAwareAdapterSize):
        super(TaskAwareWeightScheduler, self).__init__()
        self.use_mlp_for_weight = config['use_mlp_for_weight']
        self.embeddingTableSize = embeddingTableSize
        self.userAwareAdapterSize = userAwareAdapterSize

        self.taskKnowledgeNecessitySize = self.userAwareAdapterSize
        self.taskRationalitySize = self.embeddingTableSize + self.userAwareAdapterSize

        if not self.use_mlp_for_weight:
            self.taskStepSize_lstm = nn.LSTM(1, 10, 1, bidirectional=True)
            self.taskKnowledgeNecessity_lstm = nn.LSTM(self.taskKnowledgeNecessitySize, 10, 1, bidirectional=True)
            self.taskRationality_lstm = nn.LSTM(self.taskRationalitySize, 10, 1, bidirectional=True)
            input_dim = 60
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
                nn.Softmax(dim=0)
            )
        else:
            input_dim = 1 + self.embeddingTableSize + self.userAwareAdapterSize * 2
            # input_dim = 1 + self.embeddingTableSize + self.userAwareAdapterSize
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 64, bias=False),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.Tanh(),
                # nn.Linear(128, 32),
                # nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softmax(dim=0)
            )
        self.device = device

    def forward(self, batchTaskFactor):
        taskid = []
        taskStepSize = []
        taskKnowledgeNecessity = []
        taskRationality = []
        for (tid, tss, tkn, tr) in batchTaskFactor:
            taskid.append(tid)
            taskStepSize.append(tss)
            taskKnowledgeNecessity.append(tkn)
            taskRationality.append(tr)
        taskid = torch.tensor(taskid, device=self.device)           # BatchSize* 1
        taskStepSize = torch.tensor(taskStepSize, device=self.device)           # BatchSize* 1
        taskKnowledgeNecessity = torch.stack(taskKnowledgeNecessity, dim=0)     # BatchSize* 6
        taskRationality = torch.stack(taskRationality, dim=0)                   # BatchSize* 6

        # 对taskStepSize进行归一化
        taskStepSize = nn.LayerNorm(taskStepSize.shape, device=self.device)(taskStepSize)
        taskKnowledgeNecessity = self.layer_norm(taskKnowledgeNecessity)
        taskRationality = self.layer_norm(taskRationality)


        if not self.use_mlp_for_weight:
            taskStepSize_output, _ = self.taskStepSize_lstm(taskStepSize.reshape(1, len(taskStepSize), 1))
            taskStepSize_output = taskStepSize_output.sum(0)

            taskKnowledgeNecessity_output, _ = self.taskKnowledgeNecessity_lstm(taskKnowledgeNecessity.reshape(1, len(taskKnowledgeNecessity), -1))
            taskKnowledgeNecessity_output = taskKnowledgeNecessity_output.sum(0)

            taskRationality_output, _ = self.taskRationality_lstm(taskRationality.reshape(1, len(taskKnowledgeNecessity), -1))
            taskRationality_output = taskRationality_output.sum(0)

            # x = torch.cat((taskStepSize_output, taskKnowledgeNecessity_output, taskRationality_output), dim=1)
            x = torch.cat((taskStepSize_output, taskKnowledgeNecessity_output, taskRationality_output), dim=1)

            z = self.mlp(x)
        else:
            x = torch.cat((taskStepSize.unsqueeze(-1), taskKnowledgeNecessity, taskRationality), dim=1)
            # x = torch.cat((taskStepSize.unsqueeze(-1), taskRationality), dim=1)
            z = self.mlp(x)

        recorder = taskid.unsqueeze(-1), z, taskStepSize.unsqueeze(-1), torch.mean(taskKnowledgeNecessity, dim=1, keepdim=True), torch.mean(taskRationality, dim=1, keepdim=True)

        recorder = torch.stack(recorder).squeeze(-1)
        return z, recorder

    def layer_norm(self, matrix_tensor):
        # Calculate mean and variance for each row
        mean = torch.mean(matrix_tensor, dim=1, keepdim=True)
        variance = torch.var(matrix_tensor, dim=1, keepdim=True)

        # Normalize each row using layer normalization formula
        epsilon = 1e-8  # small value for numerical stability
        normalized_matrix = (matrix_tensor - mean) / torch.sqrt(variance + epsilon)

        return normalized_matrix