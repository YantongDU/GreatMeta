from collections import OrderedDict
from recbole_metarec.MetaUtils import metaQuickStart
import recbole.utils as rec_utils
import openpyxl


def initialDict(note=None):
    rstDict = OrderedDict()
    rstDict['modelName'] = '\t'
    rstDict['dataset'] = '\t'
    rstDict['mae'] = '\t'
    rstDict['rmse'] = '\t'
    rstDict['ndcg@1'] = '\t'
    rstDict['ndcg@3'] = '\t'
    rstDict['ndcg@5'] = '\t'
    rstDict['ndcg@7'] = '\t'
    rstDict['ndcg@10'] = '\t'
    rstDict['parameter_dict'] = '\t'
    rstDict['note'] = '\t' if note is None else note
    return rstDict


def record_model_result(dataset, model_name, parameter_dict, test_result, note=None):
    rstDict = initialDict(note)
    rstDict['modelName'] = model_name
    rstDict['dataset'] = dataset
    # rstDict['parameter_dict'] = rec_utils.dict2str(parameter_dict)

    rstDict.update(test_result)
    wb = openpyxl.load_workbook("cold-start.xlsx")

    sheet = wb['Sheet1']
    sheet.append(list(rstDict.values()))
    wb.save("cold-start.xlsx")


param_config = {
    'metrics': ['mae', 'mse', 'ndcg'],
    'metric_decimal_place': 4,
    'topk': [1, 3, 5, 7],
    'valid_metric': 'mse',
    'local_update_count': 5,
    'user_inter_num_interval': [40,200],
    'epochs': 200,
    'eval_args': {'group_by':'task', 'order':'RO', 'split': {'RS': [0.7,0.1,0.2]}, 'mode': 'labeled'},
    # 'meta_args': {'support_num': 20, 'query_num': 'none'},
    'meta_args': {'support_num': 'none', 'query_num': 10},
    'train_batch_size': 32,

    'use_avg_grad': False,      # False
    'use_film': True,           # True
    'use_mlp_for_weight': True,           # False
    'use_loss_instead_of_step_size': True,           # False
}

# datasetName = 'book-crossing'
# datasetName = 'ml-1m'
datasetName = 'ml-100k'

# datasetName = 'ml-100k-CTR'

# modelName = 'FOMeLU'
# modelName = 'MAMO'
# modelName = 'TaskAdMeta'
# modelName = 'GrAdMeta-0504-bak'
# modelName = 'TaNP'
# modelName = 'NLBA'
# modelNames = ['GrAdMeta']
# modelNames = ['FOMeLU']
# modelNames = ['GrAdMeta']
# modelNames = ['GrAdMeta']
modelNames = ['GrAdMeta']
# modelNames = ['FOMeLU']
# modelNames = ['MAMO']
# modelNames = ['MeLU']

note = 'baseline,调整s/q-'
# note = 'baseline'
# note = 'baseline- GrAdMeta 不对步长正则化'

for modelName in modelNames:
    config, test_result = metaQuickStart(modelName, datasetName, param_config)
    record_model_result(datasetName, modelName, config, test_result, note=note)


'''
    ModelName & DatasetName Available
    
    FOMeLU: ml-100k, ml-1m, book-crossing
    MAMO: ml-100k, ml-1m, book-crossing
    TaNP: ml-100k, ml-1m, book-crossing
    LWA: ml-100k-CTR, ml-1m-CTR, book-crossing-CTR
    NLBA: ml-100k-CTR, ml-1m-CTR, book-crossing-CTR
    MetaEmb: ml-100k-CTR, ml-1m-CTR, book-crossing-CTR
    MWUF: ml-100k-CTR, ml-1m-CTR, book-crossing-CTR
'''