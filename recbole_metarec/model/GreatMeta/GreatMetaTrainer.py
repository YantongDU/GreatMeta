# -*- coding: utf-8 -*-
"""
@Time    : 2022/9/14
@Author  : Yantong Du
@Email    : duyantong94@hrbeu.edu.cn
"""

from tqdm import tqdm
import torch
from collections import OrderedDict

from recbole.data import Interaction
from recbole.utils import FeatureSource, set_color
from recbole.utils import get_gpu_usage
from recbole_metarec.MetaTrainer import MetaTrainer
from copy import deepcopy

import numpy as np


class GreatMetaTrainer(MetaTrainer):

    def __init__(self, config, model):
        super(GreatMetaTrainer, self).__init__(config, model)
        self.lr = config['grad_meta_args']['lr']
        self.xFields = model.dataset.fields(source=[FeatureSource.USER, FeatureSource.ITEM])
        self.yField = model.RATING

        self.meta_optimiser = torch.optim.Adam(self.model.meta_model.parameters(), self.lr)

    def taskDesolve(self, task):
        spt_x, qrt_x = OrderedDict(), OrderedDict()
        for field in self.xFields:
            spt_x[field] = task.spt[field]
            qrt_x[field] = task.qrt[field]
        spt_y = task.spt[self.yField]
        qrt_y = task.qrt[self.yField]

        spt_x, qrt_x = Interaction(spt_x), Interaction(qrt_x)
        return spt_x, spt_y, qrt_x, qrt_y

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=120,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        totalLoss = torch.tensor(0.0).to(self.device)
        recorders = []
        for batch_idx, taskBatch in enumerate(iter_data):

            loss, grad, recorder = self.model.calculate_loss(taskBatch)
            new_row = np.ones((1, recorder.shape[1])) * batch_idx
            recorder = np.vstack((new_row, recorder))
            recorders.append(recorder)
            totalLoss+=loss

            # -------------- meta update --------------

            self.meta_optimiser.zero_grad()

            # set gradients of parameters manually
            for name, param in self.model.meta_model.named_parameters():
                param.grad = grad[name]
                param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            self.meta_optimiser.step()

            self.model.keepWeightParams = deepcopy(self.model.meta_model.state_dict())

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow') + ', ' +
                                          set_color('batch task avg loss: %.4f' % loss.item(), 'blue'))

        return totalLoss / (batch_idx + 1), recorders
