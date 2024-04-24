import numpy as np
from utils.logger import logger
from utils.utils import add_modal_regular_error, downsample_dataset, introduce_noisy_labels
import random
import json
# import tensorflow as tf


class Configuration(object):
    def __init__(self, configdict):
        for key in configdict:
            setattr(self, key, configdict[key])


def merge_params(params):
    allparams = {'seed': 0, 'name': None, 'settingindex': 0, 'evalevery': 100, 'mydatabag': None, 'addNoise': 0,
                 'mini_batch_size': 128, 'dim': 1, 'outdim': 1, 'actfunctype': 'relu', 'dualactfunctype': 'minrelu',
                 'actfunctypeCOVN': 'relu', 'outputactfunctype': 'linear', 'n_iterations': 10000, 'FTAonW': False,

                 'Optimizer': 'SGD', 'woconv': 0, 

                 'reconstruct_reg': 0.0, 'implicit_reg': 0.0, 'classify_reg': 1.0,
                 'embedding_dim': 32, 'subsample_ratio': 1.0, 'subsample': 1.0, 'resample': -1,

                 'learning_rate': 0.0001, 'fine_learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'tau': 0.9,
                 'label_smt': 0.0, 'noise_rho': 0.0, 'expt_ytp': 0,

                 'epsilon': 1e-5, 'n_hidlayers': 2,
                 'n_h1': 16, 'n_h2': 16, 'n_h3': 0, 'low': -1.0, 'high': 1.0, 'discretization': 200,
                 'bestk_predictions': 5,
                 'eta': 1.0, 'lagreg': 1.0, 'highorder_reg': 0.0, 'huber_delta': 0.5,
                 'n_classes': 10, 'l2reg': 0.0, 'jacoby': 1.0, 'predict_thres': 0.01, 'n_mdncomponents': 2,

                 'n_tiles': 20, 'extra_strength': False, 'n_tilings': 1, 'lta_eta': 0.1, 'sparse_dim': None,
                 'test_tiling': False, 'actfunctypeLTA': 'linear', 'actfunctypeLTAstrength': 'linear',
                 'train_bound': False, 'similarity': 'IPlusEta',
                 'lta_input_min': -1.0, 'lta_input_max': 1.0, 'outofbound_reg': 0.0,
                 'self_strength': False, 'dynamic_tiling': False, 'individual_tiling': False,
                 'coarse_n_tiles': 5, 'coarse_eta': 0.5, 'stage_one': 100000,

                 'Ptype': 'UnifConst', 'gamma': 0.99, 'lam': 0.0, 'noiselevel': 0, 'deltatype': 0,

                 'stop_grad': False, 'polydegree': 1,
                 'power': 2, 'dirname': 'resultsfile',  'flattened_d': 256}
    ''' check all feeds are in default '''
    allin = True
    problems = []
    for key in params:
        allin = allin and (key in allparams)
        if not (key in allparams):
            problems.append(key)
    if not allin:
        print(' parameter is problematic !!!!!!!!!!!!!!!! ', problems)
        exit(0)
    for key in allparams:
        if key not in params:
            params[key] = allparams[key]
    return params


def save_to_json(name, dict):
    newdict = {}
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except:
            return False
    for key in dict:
        if is_jsonable(dict[key]):
            newdict[key] = dict[key]
    with open(name, 'w') as fp:
        json.dump(newdict, fp)


class baselearner(object):
    def __init__(self, params):
        params = merge_params(params)
        ''' manually add some attributes '''
        self.mydatabag = params['mydatabag']
        self.seed = params["seed"]
        self.settingindex = params["settingindex"]
        params['datasetname'] = self.mydatabag.name
        params['dim'] = self.mydatabag.dim
        params['outdim'] = self.mydatabag.y_train.shape[1] if len(self.mydatabag.y_train.shape)>1 else 1
        params['n_classes'] = self.mydatabag.n_classes
        self.name = params['name']
        ''' use configuration to create attributes '''
        self.config = Configuration(params)
        self.loginfo = logger()
        self.save_params_json(params)
        self._reset_config()
        ''' set random seed '''
        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.config.resample > 0:
            self.mydatabag.x_train, self.mydatabag.y_train \
                = downsample_dataset(self.mydatabag.x_train, self.mydatabag.y_train, self.config.resample)

        self.mydatabag.y_train = introduce_noisy_labels(self.mydatabag.y_train, self.config.noiselevel, self.config.noise_rho)
        self.mydatabag.get_subset_label(self.mydatabag.y_train)

    def _reset_config(self):
        if 'double_circle' in self.mydatabag.name:
            self.config.low = -2.
            self.config.high = 2.
        elif 'circle' in self.mydatabag.name:
            self.config.low = -1.
            self.config.high = 1.
        elif self.mydatabag.name in ["songyeardata", "bikesharedata", "insurancemodal", "inverse_sin"]:
            self.config.low = 0.
            self.config.high = 1.
        elif self.mydatabag.name in ["mnist", "mnistfashion", "cifar10"]:
            self.config.flattened_d = 256
        print("config low and high is ============================== ", self.config.low, self.config.high)

    def _compute_prob(self, priorities):
        return np.abs(priorities)/(np.sum(np.abs(priorities)))

    def predict(self, X):
        raise NotImplementedError

    def compute_error(self, final_log=False):
        if self.mydatabag.n_classes is not None:
            self.loginfo.log_error_discrete(self.predict, self.mydatabag, final_log)
        else:
            self.loginfo.log_error(self.predict, self.mydatabag, final_log)

    def save_params_json(self, params):
        dir = 'resultsfile/'
        name = self.mydatabag.name + '_' + self.name + '_' + 'Setting_' + str(self.settingindex) + '_Params.json'
        save_to_json(dir + name, params)

    def save_results(self):
        self.loginfo.write_to_file(self.config)

    def get_all_modes(self, X):
        print(' get all modes not implemented !!!!!!!!!!!!!!!!!! ')
        raise NotImplementedError

    ''' 
    the predict function for modal regression: Implicit, MDN, KDE
    '''
    def compute_error_modal(self, final_log=False):
        self.loginfo.log_error(self.predict, self.mydatabag, final_log)
        if self.mydatabag.name in ["toy_dataset", "sin_hc"]:
            return
        add_modal_regular_error(self.mydatabag, self.get_all_modes, self.loginfo.error_dict)
