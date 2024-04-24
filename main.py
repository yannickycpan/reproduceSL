import gc
gc.collect()
import numpy as np
import random
from utils.loaddata import get_data
import sys
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_all_jsons(file):
    from agents.baselearner import save_to_json
    json_dat = open(file, 'r')
    exp = json.load(json_dat)
    json_dat.close()
    for agname in exp["agent_names"]:
        agparam, seed, n_setting, n_total_settings = get_sweep_parameters(exp[agname], 0)
        for input_n in range(n_total_settings):
            agparam, seed, n_setting, n_total_settings = get_sweep_parameters(exp[agname], input_n)
            dir = 'resultsfile/'
            #agparam = supplement_params(agparam, seed, n_setting, agname, eval_every, mini_batch_size, mydatabag)
            name = exp["datasetname"] + '_' + agname + '_' + 'Setting_' + str(n_setting) + '_Params.json'
            save_to_json(dir + name, agparam)


def get_sweep_parameters(parameters, index):
    out = {}
    accum = 1
    for key in sorted(parameters):
        if key == 'name' or key == 'dim':
            out[key] = parameters[key]
            continue
        num = len(parameters[key])
        out[key] = parameters[key][int((index / accum) % num)]
        accum *= num
    n_run = int(index / accum)
    n_setting = int(index % accum)
    return out, n_run, n_setting, accum


def enumeratesettings(agname, params, index):
    parammat = []
    for i in range(index):
        paramvec = []
        out, _, _, _ = get_sweep_parameters(params, i)
        for key in sorted(out):
            paramvec.append(out[key])
        paramvec = np.array(paramvec)
        parammat.append(paramvec.reshape((-1, paramvec.shape[0])))
    parammat = np.vstack(parammat)
    filename = ''
    for key in sorted(params):
        filename = filename + key + '_'
    np.savetxt(agname + '_' + filename + '.txt', parammat, fmt='%10.7f', delimiter=',')


def get_learner(params, agname):
    #if agname in ['L2', 'IncrementalL2', 'HuberRegression', 'NNPoissonRegression', 'PrioritizedL2',
    #              'FullPrioritizedL2', 'FullPrioritizedL2-RecBuffer', 'FullPrioritizedL2-PBuffer']:
    #    from agents.regularregressions import nnlearner
    #    return nnlearner(params)
    if agname in ['TDLinearPoi', 'TDLinearReg', 'TDLinearCla', 'LinearPoi-WP', 'LinearReg-WP', 'LinearCla-WP',
                    'LinearPoi', 'LinearReg', 'LinearCla']:
        from agents.TDstf import tdlineartf
        return tdlineartf(params)
    elif agname in ['TDPoissonNN', 'TDRegNN', 'TDClassifyFcNN', 'TDMiniCNN', 'TDCNN3Layers', 'TDRes20',
                    'CNN3Layers',  'MiniCNN', 'Res20','PoissonNN', 'RegNN', 'ClassifyFcNN',
                    'MiniCNN-WP', 'Res20-WP', 'PoissonNN-WP', 'RegNN-WP', 'ClassifyFcNN-WP', 'CNN3Layers-WP']:
        from agents.TDstf import tdnns
        return tdnns(params)
    elif agname == 'TDGLM': # non tf version
        from agents.TDs import tdglm
        return tdglm(params)
    elif agname in ['linearclassify']:
        from agents.linearclassifier import image_linearclassifier
        return image_linearclassifier(params)
    elif agname in ['L2-LTA']:
        from agents.ltalearners import ltaregression
        return ltaregression(params)
    elif agname in ['LinearReg', 'LinearPoissonReg', 'PrioritizedLinearReg', 'FullPrioritizedLinearReg']:
        from agents.regularregressions import linear_regression
        return linear_regression(params)
    elif agname in ['Implicit', 'Implicit-Dot']:
        from agents.modalregressions import nnlearner_implicit
        return nnlearner_implicit(params)
    elif agname in ['ImplicitClassify']:
        from agents.implicitclassifier import implicit_image_classifier
        return implicit_image_classifier(params)
    elif agname == 'KDE':
        from agents.notflearners import kde_regression
        return kde_regression(params)
    elif 'Power' in agname:
        from agents.regularregressions import generalpowerlearner
        return generalpowerlearner(params)
    elif agname == 'MDN':
        from agents.modalregressions import mdnlearner
        return mdnlearner(params)
    elif agname in ['LeNet5', 'LeNet5-Res', 'LeNet5-DualRelu', 'LeNet5-DualRelu-Concat']:
        from agents.imageclassifier import image_classifier
        return image_classifier(params)
    elif agname in ['DynaPred']:
        from agents.ndtclassifier import image_dynamicclassifier
        return image_dynamicclassifier(params)
    elif agname in ['FTALinear']:
        from agents.linearfta import ftalinearlearner
        return ftalinearlearner(params)
    elif agname in ['FTAWLinear']:
        from agents.linearfta import ftawlinearlearner
        return ftawlinearlearner(params)
    elif agname in ['FTARegression']:
        from agents.ltalearners import ltaregression
        return ltaregression(params)
    elif agname in ['SmallW']:
        from agents.smallw import image_classifier_smallw
        return image_classifier_smallw(params)
    elif agname in ['LeNet5-LTA', 'LeNet5-MidLTA']:
        from agents.ltalearners import image_classifier_lta
        return image_classifier_lta(params)
    else:
        print(' **************************** learner not found **************************** ')
        exit(0)


def supplement_params(agentparams, n, nsetting, name, evalevery, n_iterations, mydatabag):
    if 'seed' not in agentparams:
        agentparams['seed'] = n
    if 'name' not in agentparams:
        agentparams['name'] = name
    if 'settingindex' not in agentparams:
        agentparams['settingindex'] = nsetting
    if 'evalevery' not in agentparams:
        agentparams['evalevery'] = evalevery
    agentparams['n_iterations'] = n_iterations
    #if 'mini_batch_size' not in agentparams:
    #    agentparams['mini_batch_size'] = mini_batch_size
    agentparams['mydatabag'] = mydatabag
    return agentparams


def find_dataname(exp, input_n):
    maxs = 0
    for agname in sorted(exp["agent_names"]):
        _, _, _, n_total_settings = get_sweep_parameters(exp[agname], 0)
        if n_total_settings > maxs:
            maxs = n_total_settings
    maxs = exp["nruns"] * maxs
    datanameind = int(input_n / maxs)
    print(' dataname id is ============== ', datanameind)
    count = 0
    for dataname in sorted(exp["datasetname"]):
        if count == datanameind:
            input_n = input_n - (maxs * count)
            return dataname, int(input_n)
        count += 1
    print(' index out of total number of runs ======================== ')
    exit(0)


if __name__ == '__main__':

    file = sys.argv[1]
    input_n = int(sys.argv[2])
    json_dat = open(file, 'r')
    exp = json.load(json_dat)
    json_dat.close()

    n_runs_per_setting = exp["nruns"]
    n_iterations = exp["niterations"]
    eval_every = exp["evalEverySteps"]
    datasetname, input_n = find_dataname(exp, input_n)
    datadir = exp["datadir"]

    print(' the dataset name and adjusted input n is :: ========== ', datasetname, input_n)

    '''each n indicates one run'''
    learners = {}
    mydatabag = None
    for agname in sorted(exp["agent_names"]):
        agparam, seed, n_setting, n_total_settings = get_sweep_parameters(exp[agname], input_n)
        if input_n >= n_total_settings * n_runs_per_setting:
            print('the input index is NOT applicable to current agent ' + agname)
            continue
        '''seed will not be changed by algo which is not applicable'''
        np.random.seed(seed)
        random.seed(seed)
        # if mydatabag is None:
        addnoise = 1 if 'addNoise' not in agparam else agparam['addNoise']
        subsample = agparam["subsample"] if "subsample" in agparam else 1.0

        mydatabag = get_data(datasetname, datadir, addnoise, subsample)
        ''' add some to agent params '''
        supplement_params(agparam, seed, n_setting, agname, eval_every, n_iterations, mydatabag)
        print('for agent ', agname, ' use parameter setting: ------------- ', n_setting, seed, agparam)
        learners[agname] = get_learner(agparam, agname)

    # x_train, y_train, x_test, y_test = mydatabag.x_train, mydatabag.y_train, mydatabag.x_test, mydatabag.y_test

    for i in range(n_iterations):
        if i % eval_every == 0 or i == n_iterations-1:
            for key in learners:
                print('current compute error ----------------------- :: ', key)
                '''handle writing results for circle example'''
                learners[key].compute_error()
                learners[key].save_results()
        for key in learners:
            mini_batch_size = min(learners[key].mydatabag.x_train.shape[0], learners[key].config.mini_batch_size)
            mini_batch_inds = random.sample(range(learners[key].mydatabag.x_train.shape[0]), k=mini_batch_size)
            # print(mini_batch_inds)
            learners[key].train(learners[key].mydatabag.x_train[mini_batch_inds, :],
                                learners[key].mydatabag.y_train[mini_batch_inds])

    for key in learners:
        # in the end, compute the error on the training set
        learners[key].compute_error(True)
        learners[key].save_results()
