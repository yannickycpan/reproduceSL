import numpy as np
from scipy.spatial.distance import directed_hausdorff


""" classification tasks """


def classify_error(labels, predictions, multiclass):
    if multiclass:
        labels = np.argmax(labels, axis=1)
    else:
        labels = np.squeeze(labels)
    return np.sum(labels != predictions)/float(len(labels))


def get_subset_train(mydatabag, maxn=50000):
    if mydatabag.x_train.shape[0] <= maxn:
        return mydatabag.x_train, mydatabag.y_train
    randsubset = np.random.randint(0, mydatabag.x_train.shape[0], maxn)
    x_train, y_train = mydatabag.x_train[randsubset, :], mydatabag.y_train[randsubset]
    return x_train, y_train


""" 
used for modal regression algs
"""


def get_local_prediction_finite_diff(lossvec, delta, Y, epsilon=1e-3):
    first_der, second_der = finite_diff_der(lossvec, delta)

    ''' for convenience, use finite difference method to check first and second order condition '''
    boolsubset = (np.abs(first_der) < epsilon) * (second_der > 0)

    subY = Y[1:-1]
    if np.sum(boolsubset) == 0:
        boolsubset[np.argmin(lossvec[1:-1])] = True
    avgmodes_der = np.sum(boolsubset)
    predictions = subY[boolsubset]
    return predictions, avgmodes_der


def get_local_prediction(lossvec, Y):
    subY = Y[1:-1]
    selfloss = lossvec[1:-1]
    ''' for each point, compute its neighbors' loss '''
    leftloss = lossvec[:-2]
    rightloss = lossvec[2:]
    boolsubset = (leftloss - selfloss > 0) * (rightloss - selfloss > 0)
    if np.sum(boolsubset) == 0:
        boolsubset[np.argmin(selfloss)] = True
    avgmodes_der = np.sum(boolsubset)
    predictions = subY[boolsubset]
    return predictions, avgmodes_der


def get_global_prediction(lossvec, Y, epsilon):
    minset_pred = np.abs(np.min(lossvec) - lossvec) < epsilon
    predictions_minset = Y[minset_pred]
    avgmodes_minset = np.sum(minset_pred)
    return predictions_minset, avgmodes_minset


def list_to_numpy_mat(arrays, maskval=0.0):
    maxlen = 1
    npmat = np.ones((len(arrays), 400)) * maskval
    for i, ar in enumerate(arrays):
        if ar.shape[0] > maxlen:
            maxlen = ar.shape[0]
        npmat[i, :ar.shape[0]] = ar
    return npmat[:, :maxlen]


def finite_diff_der(lossvec, delta):
    first_der = (lossvec[2:] - lossvec[:-2]) / delta
    second_der = (lossvec[2:] + lossvec[:-2] - 2. * lossvec[1:-1]) / delta ** 2
    return first_der, second_der


def compute_rmse_one_pred_multi_target(truetargets, predictions):
    diff = np.abs(truetargets - predictions.reshape((-1, 1)))
    inds = np.nanargmin(diff, axis=1)
    truetargets = truetargets[np.arange(truetargets.shape[0]), inds]
    return compute_rmse(truetargets, predictions)


def compute_mae_one_pred_multi_target(truetargets, predictions):
    diff = np.abs(truetargets - predictions.reshape((-1, 1)))
    inds = np.nanargmin(diff, axis=1)
    truetargets = truetargets[np.arange(truetargets.shape[0]), inds]
    return compute_mae(truetargets, predictions)


def compute_rmse_mae_multi_targets(targets, predictions):
    rmse = compute_rmse_one_pred_multi_target(targets, predictions)
    mae = compute_mae_one_pred_multi_target(targets, predictions)
    return rmse, mae


def add_rmse_mae_multi_targets(targets, predictions, error_dict, prefix='test'):
    rmse, mae = compute_rmse_mae_multi_targets(targets, predictions)
    error_dict[prefix + '-rmse'].append(rmse)
    error_dict[prefix + '-mae'].append(mae)
    print(' rmse and mae in updated repo are ============================= ', rmse, mae)


def add_modal_regression_validation_error(dataX, dataY, predictmodes, error_dict, key='valid-rmse'):
    """
    calculate the closest prediction to the given target in the training/validation set
    multi-prediction, single target
    """
    predictions_local, nummodes_local, predictions_global, nummodes_global = predictmodes(dataX)
    error_dict['local-num-modes-valid'].append(nummodes_local)
    error_dict['global-num-modes-valid'].append(nummodes_global)

    ''' when compute rmse; swap the prediction and target to use multi target '''
    rmse_global = compute_rmse_one_pred_multi_target(predictions_global, dataY)
    rmse_local = compute_rmse_one_pred_multi_target(predictions_local, dataY)

    error_dict['global-' + key].append(rmse_global)
    error_dict['local-' + key].append(rmse_local)
    print('best validation error of local and global set is :: ================ ', rmse_local, rmse_global)


def modal_prediction_yhat_log(mydatabag, agname, predictmodes):
    if mydatabag.name in ["circle", "biased_circle"]:
        x = np.arange(-1.0, 1.0, 0.01)
    elif mydatabag.name == "inverse_sin":
        x = np.arange(0.0, 1.0, 0.01)
    else:
        x = np.arange(-2.0, 2.0, 0.01)
    predictions_local, _, predictions_global, _ = predictmodes(x)
    fname = mydatabag.name + '_' + agname + '_' + 'yhat-allmodes_local.txt'
    np.savetxt(fname, predictions_local, fmt='%10.7f', delimiter=',')

    fname = mydatabag.name + '_' + agname + '_' + 'yhat-allmodes_global.txt'
    np.savetxt(fname, predictions_global, fmt='%10.7f', delimiter=',')


def get_hausdorff_dist(y, predictions):
    dist = np.zeros(predictions.shape[0])
    if len(y.shape) < 2:
        y = y[:, None]
    if len(predictions.shape) < 2:
        predictions = predictions[:, None]
    for id in range(predictions.shape[0]):
        u = predictions[id, :]
        u = u[~np.isnan(u)].reshape((-1, 1))
        v = y[id, :]
        v = v[~np.isnan(v)].reshape((-1, 1))
        hausdorff_dist = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        dist[id] = hausdorff_dist
    return np.mean(dist), np.sqrt(np.mean(np.power(dist, 2)))


def add_modal_regular_error(mydatabag, predictmodes, error_dict):
    dataX, dataY = mydatabag.x_test, mydatabag.y_test_true_modes

    predictions_local, nummodes_local, predictions_global, nummodes_global = predictmodes(dataX)
    predictions_local = mydatabag.inv_transform(predictions_local)
    predictions_global = mydatabag.inv_transform(predictions_global)

    error_dict['local-num-modes'].append(nummodes_local)
    error_dict['global-num-modes'].append(nummodes_global)

    """ compute hausdorff dist """
    mae_local, rmse_local = get_hausdorff_dist(dataY, predictions_local)
    mae_global, rmse_global = get_hausdorff_dist(dataY, predictions_global)

    error_dict['local-hausdorff-mae'].append(mae_local)
    error_dict['local-hausdorff-rmse'].append(rmse_local)

    error_dict['global-hausdorff-mae'].append(mae_global)
    error_dict['global-hausdorff-rmse'].append(rmse_global)
    print(' predicted local and global rmse are ======================= ', rmse_local, rmse_global)
    print(' predicted local and global mae are ======================= ', mae_local, mae_global)


"""
general utils
"""


def save_to_file(name, error):
    error.tofile(name, sep=',', format='%10.7f')


def compute_rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(np.squeeze(targets) - np.squeeze(predictions))))


def compute_mae(targets, predictions):
    return np.mean(np.abs(np.squeeze(targets) - np.squeeze(predictions)))


def add_error_general(targets, predictions, error_dict, prefix='test'):
    rmse = compute_rmse(targets, predictions)
    mae = compute_mae(targets, predictions)
    error_dict[prefix + '-rmse'].append(rmse)
    error_dict[prefix + '-mae'].append(mae)
    print('rmse and mae are :: -------------- ', rmse, mae)


def get_rmse_sin(y_test, x_test, predictions, error_dict, prefix='test'):
    testrmse = compute_rmse(y_test, predictions)
    testrmsehigh = compute_rmse(y_test[x_test[:, 0] < 0], predictions[x_test[:, 0] < 0])
    testrmselow = compute_rmse(y_test[x_test[:, 0] >= 0], predictions[x_test[:, 0] >= 0])
    error_dict[prefix+'-rmse'].append(testrmse)
    print(' the rmse is ------------------------------------------------- ', testrmse)
    error_dict[prefix+'-high'].append(testrmsehigh)
    error_dict[prefix+'-low'].append(testrmselow)