import os
from utils.utils import *
import math


class logger:
    def __init__(self):
        ''' full obj now means the one predicted by randomly pick up a mode from the set of minimas '''
        self.error_dict = {'train-rmse':[],'test-rmse':[], 'train-mae':[], 'test-mae':[],
                           'test': [], 'train': [], 'valid':[], 'wtest':[],

                           'test-TD': [], 'train-TD': [], 'valid-TD': [],
                           'test-b-Aw': [], 'train-b-Aw': [], 'valid-b-Aw':[],
                           'wdist-TD': [], 'wdist-b-Aw': [],

                           'train-high':[], 'test-high':[], 'train-low':[], 'test-low':[],

                           'local-hausdorff-mae':[], 'local-hausdorff-rmse':[],
                           'global-hausdorff-mae':[], 'global-hausdorff-rmse':[],
                           'hausdorff-mae': [], 'hausdorff-rmse': [],

                           'local-num-modes-valid':[], 'global-num-modes-valid':[],
                           'local-num-modes': [], 'global-num-modes': [],

                           'yhat':[], 'yhat-allmodes': [], 'loss': [], 'fxy': [], 'test-rmse-unique': [],

                           # used for average full-subset
                           'avgdistratio-fullset': [], 'avgdistratio-subset': [],

                           'grad-norm': [], 'instance-sparse-midphi': [],

                           'instance-sparse': [], 'overlap-sparse': [],
                           }

    def save_all_params_to_json(self, name, dict):
        import json
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

    def write_to_file(self, config):
        datasetname, agname, seed, settingindex, dirname \
            = config.datasetname, config.name, config.seed, config.settingindex, config.dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for key in self.error_dict:
            if len(self.error_dict[key]) > 0:
                save_to_file(dirname + '/' + datasetname + '_' + agname + '_' + key
                             + '_setting_' + str(settingindex) + '_run_' + str(seed)
                             + '_LC.txt', np.array(self.error_dict[key]))

    def log_error(self, predict, mydatabag, final_log = False):
        x_train, y_train, x_test, y_test = mydatabag.x_train, mydatabag.y_train, mydatabag.x_test, mydatabag.y_test
        ''' DO not use double predictions for single-valued datasets '''
        if "toy_dataset" in mydatabag.name or "biased_sin" in mydatabag.name or "sin_hc" in mydatabag.name:
            predictions = predict(x_test)
            get_rmse_sin(y_test, x_test, predictions, self.error_dict)
            if "sin_hc" in mydatabag.name:
                predictions = predict(x_train)
                get_rmse_sin(y_train, x_train, predictions, self.error_dict, 'train')
            if final_log: # this was actually for toy dataset
                x = np.arange(-2.5, 2.51, 1 / 32)
                x = x.reshape((-1, 1))
                predictions = predict(x)
                self.error_dict['yhat'] = predictions
        elif mydatabag.name in ["circle", "double_circle", "inverse_sin",
                                "high_double_circle", "biased_circle", "insurancemodal"]:
            predictions = predict(x_test)
            predictions = mydatabag.inv_transform(predictions)
            add_rmse_mae_multi_targets(mydatabag.y_test_true_modes, predictions, self.error_dict)
            mae, rmse = get_hausdorff_dist(mydatabag.y_test_true_modes, predictions)
            self.error_dict['hausdorff-mae'].append(mae)
            self.error_dict['hausdorff-rmse'].append(rmse)
            """ for biased circle, think about how to get mode with lower likelihood """
            if mydatabag.name in ["biased_circle"]:
                predictions = predict(x_test)
                rmse = compute_rmse(mydatabag.most_likely_mode, predictions)
                self.error_dict['test-rmse-unique'].append(rmse)
                print(' biased circle rmse is ====================== ', rmse)
        elif mydatabag.name == "toy_yaoli":
            predictions = predict(x_test)
            predictions = mydatabag.inv_transform(predictions)
            truetargets = 2.0 * np.sin(math.pi * np.squeeze(x_test)) + 1.0 + 2.0 * np.squeeze(x_test)
            testrmse = compute_rmse(truetargets, np.squeeze(predictions))
            self.error_dict['test-rmse'].append(testrmse)
            print(' toy yaoli test rmse is :: ================ ', testrmse)
        else:
            predictions = predict(x_test)
            predictions = mydatabag.inv_transform(predictions)
            print(' the test error is :: ================================== ')
            add_error_general(y_test, predictions, self.error_dict, prefix='test')

            x_train, y_train = get_subset_train(mydatabag)
            predictions = predict(x_train)
            predictions = mydatabag.inv_transform(predictions)
            original_y_train = mydatabag.inv_transform(y_train)
            print(' the train error is :: ================================== ')
            add_error_general(original_y_train, predictions, self.error_dict, prefix='train')


    def log_error_discrete(self, predict, mydatabag, final_log=False):
        # if final_log:
        
        if mydatabag.x_train.shape[0] < 10000:
            subtrainx = mydatabag.x_train
            subtrainy = mydatabag.y_train
            predictions = predict(subtrainx)
        else:
            inds = np.random.randint(0, mydatabag.x_train.shape[0], 10000)
            subtrainx = mydatabag.x_train[inds, :]
            subtrainy = mydatabag.y_train[inds, :]
            predictions = predict(subtrainx)
        trainerr = classify_error(subtrainy, predictions, True)
        self.error_dict['train'].append(trainerr)
        
        predictions = predict(mydatabag.x_test)
        testerr = classify_error(mydatabag.y_test, predictions, True)
        self.error_dict['test'].append(testerr)

        if mydatabag.y_test.shape[1] == 2:
            weighted_testerr = classify_error_reweighting(mydatabag.y_test, predictions)
            self.error_dict['wtest'].append(weighted_testerr)
            print('test, wtest, train, classification error is --------------------------- ', testerr, weighted_testerr, trainerr)
        else:
            print('test, train, classification error is --------------------------- ', testerr, trainerr)

        if mydatabag.x_valid is not None:
            x_valid, y_valid = mydatabag.x_valid, mydatabag.y_valid
            predictions = predict(x_valid)
            validerr = classify_error(mydatabag.y_valid, predictions, True)
            self.error_dict['valid'].append(validerr)
            print('validation, test, train, classification error is --------------------------- ',
                  validerr, testerr, trainerr)

    def log_error_vae(self, predict, mydatabag, final_log=False):
        error = predict(mydatabag.x_test)
        self.error_dict['test'].append(error)
        print('image reconstruction error is -------------------- ', error)

    def log_sparsity(self, instance_sparsity, overlap_sparsity):
        self.error_dict['instance-sparse'].append(instance_sparsity)
        self.error_dict['overlap-sparse'].append(overlap_sparsity)
        print('instance and overlap sparsity are:: -------------------- ', instance_sparsity, overlap_sparsity)

