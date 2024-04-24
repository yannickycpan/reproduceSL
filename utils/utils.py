import numpy as np
from scipy.spatial.distance import directed_hausdorff
import scipy

def compute_exptytp(P, y):
    if isinstance(P, float):
        if len(y.shape) == 2 and y.shape[1]>1:
            return np.mean(convert2linear('mnist', y), axis=0)
        return np.mean(y, axis=0)
    elif len(y.shape) == 2 and P.shape[0] == y.shape[1]:
        return P * convert2linear('mnist', np.ones((1, y.shape[1]))) # all classification uses the same conversion
    elif P.shape[0] == y.shape[0]: # reg task, when P is mat
        return np.squeeze(P.dot(y.reshape((y.shape[0], -1))))
    else:
        print(' there is error in the P type !!! ')
        exit(0)

def downsample_dataset(X, Y, p):
    num_samples = X.shape[0]
    num_zeros, num_ones = np.sum(Y, axis=0)
    new_num_ones = int(p * num_zeros/(1 - p))
    # Indices of ones and zeros in the dataset
    ones_indices = np.where(np.argmax(Y, axis=1) == 1)[0]
    zeros_indices = np.where(np.argmax(Y, axis=1) == 0)[0]

    # Randomly select indices for downsampling
    selected_ones_indices = np.random.choice(ones_indices, new_num_ones, replace=False) \
        if num_ones/num_samples > p else ones_indices

    # Concatenate the selected indices
    selected_indices = np.concatenate((selected_ones_indices, zeros_indices))

    # Shuffle the indices to maintain randomness
    np.random.shuffle(selected_indices)

    # Create the downsampled dataset
    downsampled_X = X[selected_indices]
    downsampled_Y = Y[selected_indices]

    return downsampled_X, downsampled_Y

def softmax(z):
    if len(z.shape) < 2:
        z = z.reshape((1, -1))
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    # exp_z = np.exp(z)
    prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    # return np.where(prob>=0.5, 1., 0.)
    return prob

def sigmoid(x):
    return 1./(1. + np.exp(-x))
    # return np.where(x>=1e-5, 1., 0.)

def convert2linear(name, x):
    if name in ['mnist', 'mnistfashion', 'cifar10', 'cifar100', 'heart', "ausweather", "travelinsurance", "cancer", "ionosphere"]:
        x = np.clip(x, 1e-15, 1. - 1e-15)  # Clip x to avoid division by zero
        return np.log(x / (1. - x))
    elif name in ['bikesharedata']:
        return np.log(np.maximum(x, 0.) + 1.)
    return x

def backfromlinear(name, x):
    if name in ['mnist', 'mnistfashion', 'cifar10', 'cifar100', 'heart', "ausweather", "travelinsurance", "cancer", "ionosphere"]:
        return softmax(x)
    elif name in ['bikesharedata']:
        return np.exp(x) - 1.
    return x

def get_seq_noise(noiselevel, noise_rho, n):
    noises = [np.random.normal(0., 1.) * noiselevel]
    for i in range(n-1):
        noises.append(np.random.normal(noise_rho * noises[-1], 1. - noise_rho**2) * noiselevel)
    return np.array(noises)

def gen_pos_cov(n, c):
    # block = np.ones([int(40), int(40)]) * c
    # block_list = [block for i in range(int(n/40+1))]
    # cov = scipy.linalg.block_diag(*block_list)
    cov = (np.ones((n, n)) - np.eye(n))*c + np.eye(n)
    return cov

def get_cor_noise(n, c):
    cov = gen_pos_cov(n, c)
    noise = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov) 
    noise = noise[:n]
    return noise

def introduce_noisy_labels(y_train, noiselevel, noise_rho):
    if noiselevel == 0:
        return y_train
    elif len(y_train.shape)==1 or y_train.shape[1] == 1:
        if noise_rho == 0:
            noise = np.random.normal(0., 1., y_train.shape) * noiselevel * 5.
        else:
            noise = get_seq_noise(noiselevel, noise_rho, y_train.shape[0])
        return y_train + noise.reshape(y_train.shape)
    num_samples, num_classes = y_train.shape

    # Calculate the number of labels to flip
    num_flips = int(num_samples * noiselevel * num_classes/(num_classes-1.))
    # Generate random indices to flip labels
    flip_indices = np.random.choice(num_samples, num_flips, replace=False)

    # Flip the labels at the selected indices
    noisy_y_train = y_train.copy()
    wrongls = np.arange(num_classes)
    np.random.shuffle(wrongls)
    for idx in flip_indices:
        if noise_rho > 0:
            l = np.argmax(y_train[idx])
            noisy_label = wrongls[l]
        else:
            noisy_label = np.random.choice(num_classes, 1)[0]
        noisy_y_train[idx] = np.eye(num_classes)[noisy_label]  # Create a one-hot vector
    return noisy_y_train

''' rho is correlation coefficient, set to zero becomes i.i.d. gaussian;
    this func is not for classification tasks '''
def get_cond_noise(name, curnoise, noiselevel, n, noise_rho=0.9):
    if name in ['mnist', 'mnistfashion', 'cifar10', 'cifar100',
                'heart', "ausweather", "travelinsurance", "cancer", "ionosphere"]:
        return 0.
    return np.random.normal(noise_rho * curnoise, 1. - noise_rho**2, n) * noiselevel

def convert2onehot(nparr):
    onehoty = np.zeros((len(nparr), len(np.unique(nparr))))
    onehoty[np.arange(len(nparr)), nparr.astype(int)] = 1.0
    return onehoty

def rmse(hatv, v):
    return np.sqrt(np.mean(np.square(hatv - v)))

def acc(ypred, ytrue):
    return np.sum(np.argmax(ypred,axis=1) != np.argmax(ytrue,axis=1))/ypred.shape[0]

def compute_error4tds(loginfo, x_train, y_train, x_test, y_test, W, lrw, suffix, name=None):
    yhattrain = np.squeeze(x_train.dot(W))
    yhattest = np.squeeze(x_test.dot(W))
    if name == 'bikesharedata':
        yhattest = np.exp(yhattest) - 1.
        yhattrain = np.exp(yhattrain) - 1.
    if W.shape[1]>1:
        loginfo.error_dict['train-' + suffix].append(acc(yhattrain, y_train))
        loginfo.error_dict['test-' + suffix].append(acc(yhattest, y_test))
    else:
        loginfo.error_dict['train-'+suffix].append(rmse(yhattrain, y_train))
        loginfo.error_dict['test-'+suffix].append(rmse(yhattest, y_test))

    if lrw is not None:
        loginfo.error_dict['wdist-'+suffix].append(np.linalg.norm(W - lrw))
    print(' td test err is --------------------------- ', loginfo.error_dict['test-'+suffix][-1])

def compute_Ddiag(P):
    evals, evecs = np.linalg.eig(P.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:, 0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real
    return stationary

def compute_A_b_mspbe(X, y, P, gamma, lam=0.):
    inv = np.linalg.inv(np.eye(P.shape[0])-gamma*lam*P) if lam != 0 else 1.0
    rpi = y - gamma * P.dot(y)
    try:
        Ddiag = compute_Ddiag(P).reshape((1, -1))
    except:
        Ddiag = 1.
        print(' error in computing Dpi happened !!!!!!!!!!!!!!!!!!!!!!!!!!!! ')
    bpi = (X.T * Ddiag).dot(inv).dot(rpi)
    precondi = (X.T * Ddiag).dot(inv).dot(np.eye(P.shape[0]) - gamma * P)
    Amat = precondi.dot(X)
    return Amat, bpi, rpi

def getP(Ptype, name, X, Y=None, epsilon=0.):
    if Ptype == 'NGram':
        return transP.ngram(X)
    elif Ptype == 'Gram':
        return transP.gram(X)
    elif Ptype == 'DSMGram':
        return transP.dsm_gram(X)
    elif Ptype == 'DSMYsim':
        return transP.dsm_ysim(Y, epsilon, name)
    elif Ptype == 'DSMYdist':
        return transP.dsm_ydist(Y, epsilon, name)
    elif Ptype == 'DSMXsim':
        return transP.dsm_xsim(X, epsilon)
    elif Ptype == 'FastP':
        return transP.fastp(X)
    elif Ptype == 'UnifConst':
        return transP.uniformconstP(X)
    elif Ptype == 'UnifRand':
        return transP.uniformrandP(X)
    else:
        print(' P type not available')
        exit(0)

def sample_tp(mydatabag, config, P, cur_inds):
    # for classification, DSMYsim and DSMYdist are the same
    if config.n_classes is not None and config.Ptype == 'DSMYsim':
        ylabel = sample_from_P(P, mydatabag.x_train.shape[0],
                               np.argmax(mydatabag.y_train[cur_inds], axis=1), config.mini_batch_size)
        ids = []
        for yl in ylabel:
            ids.append(np.random.choice(a=mydatabag.label_to_samples[yl], size=1)[0])
        return ids
    else:
        return sample_from_P(P, mydatabag.x_train.shape[0], cur_inds, config.mini_batch_size)

# input a transition probaility matrix, sample the next state index
def sample_from_P(P, Pdim, indt, n=1):
    if isinstance(P, float):
        return np.random.randint(0, Pdim, n)
    elif len(P.shape)==1: # prob vec, used only for fastP
        if P[indt] == indt:
            return np.random.choice(P.shape[0])
        else:
            return P[indt]
    if n > 1:
        ids = []
        for i in range(n):
            ids.append(np.random.choice(a=P.shape[0], p=np.squeeze(P[indt[i],:]), size=1)[0])
        return ids
    return np.random.choice(a=P.shape[0], p=np.squeeze(P[indt,:]), size=1)


def get_fastsimP(X):
    n_rows, n_cols = X.shape
    arrres = np.zeros(n_rows, dtype=int)

    # Loop through each row
    for i in range(n_rows):
        # Calculate the similarity between the current row and all other rows
        similarities = np.linalg.norm(X - X[i], axis=1)
        # Exclude the current row and rows already assigned to others
        similarities[i] = np.inf
        similarities[arrres[:i]] = np.inf
        if np.all(similarities==np.inf):
            most_similar_index = i
        else:
            # Find the index of the smallest similarity
            most_similar_index = np.argmin(similarities)
        # Mark the most similar row as 1 in the result matrix
        arrres[i] = most_similar_index
    return arrres

def gaussian_kernel_matrix(X):
    # Compute the pairwise squared Euclidean distances between all pairs of rows in X
    pairwise_distances = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    # Calculate the median squared distance
    median_squared_distance = np.median(pairwise_distances)
    # Compute gamma using the heuristic: gamma = 1 / (2 * median_squared_distance)
    gamma = 1.0 / (2.0 * median_squared_distance)
    # Compute the Gaussian kernel matrix using the computed gamma
    kernel_matrix = np.exp(-gamma * pairwise_distances)
    return kernel_matrix

class transP(object):
    @staticmethod
    def gram(X):
        return X.dot(X.T)

    @staticmethod
    def ngram(X):
        if np.any(X<0):
            X += np.max(np.abs(X))
        P = X.dot(X.T)
        return P / np.sum(P, axis=1).reshape((-1, 1))

    @staticmethod
    def dsm_gram(X):
        return dsm_projection(transP.gram(X))

    @staticmethod
    def fastp(X):
        return get_fastsimP(X)

    @staticmethod
    def dsm_ysim(Y, eps, name=None):
        if name in ['mnist', 'mnistfashion', 'cifar10', 'cifar100', "heart", "ausweather", "travelinsurance", "cancer", "ionosphere"]:
            nc = Y.shape[1]
            P = np.eye(nc) *  eps
            offdiag = (np.ones((nc, nc)) - np.eye(nc)) * (1.-eps)/(nc-1.)
            return P + offdiag
        Y = np.squeeze(Y)
        diff = Y[:, None]-Y[None,:]
        nvar = np.var(Y)/Y.shape[0]
        sqdiff = np.square(diff)
        ysim = np.exp(-sqdiff/(nvar + 1e-5)) + eps
        itk = 200 if Y.shape[0] > 5000 else 1000
        return dsm_projection(ysim, itk)

    @staticmethod
    def dsm_ydist(Y, eps, name=None):
        if name in ['mnist', 'mnistfashion', 'cifar10', 'cifar100', "heart", "ausweather", "travelinsurance", "cancer",
                    "ionosphere"]:
            nc = Y.shape[1]
            P = np.eye(nc) * eps
            offdiag = (np.ones((nc, nc)) - np.eye(nc)) * (1. - eps) / (nc - 1.)
            return P + offdiag
        Y = np.squeeze(Y)
        diff = Y[:, None]-Y[None,:]
        nvar = np.var(Y)/Y.shape[0]
        sqdiff = np.square(diff)
        ydist = 1.0 - np.exp(-sqdiff/(nvar + 1e-5)) + eps
        itk = 200 if Y.shape[0] > 5000 else 1000
        return dsm_projection(ydist, itk)

    @staticmethod
    def dsm_xsim(X, eps, name=None):
        Kmat = gaussian_kernel_matrix(X) + eps
        itk = 200 if X.shape[0] > 5000 else 1000
        return dsm_projection(Kmat, itk)

    @staticmethod
    def dsm(X, eig_sign='pos'):
        return vectorized_constr_ds_matrix(X.shape[0], eig_sign)

    @staticmethod
    def uniformrandP(X, up = 100.):
        P = np.random.uniform(0., up, (X.shape[0], X.shape[0]))
        return P / np.sum(P, axis=1).reshape((-1, 1))

    @staticmethod
    def gaussianrandP(X, scale=100.):
        P = np.abs(np.random.normal(0., scale, (X.shape[0], X.shape[0])))
        return P / np.sum(P, axis=1).reshape((-1, 1))

    @staticmethod
    def uniformconstP(X):
        return 1./X.shape[0]


def matrix_entries(k, l, spectrum):
    # k and l: int, it should range (0, spectrum.size)
    # spectrum: np.array, spectrum sigma = (1, lambda_2, ..., lambda_n)
    # Output: float, returns the element of a matrix on the kth row and lth column.
    n = spectrum.size
    A = [np.sin(2 * np.pi * k * j / n + np.pi / 4) * np.sin(2 * np.pi * l * j / n + np.pi / 4)
         for j in range(1, n)]
    return (1. / n) * (1 + 2 * np.dot(spectrum[1:], A))


""" This function checks if the spectrum generates a valid symmetric doubly stochastic matrix"""


def verify_if_mat_valid(spectrum):
    # spectrum: np.array, spectrum sigma = (1, lambda_2, ..., lambda_n)
    # Output: bool, True or False depending if the spectrum
    #         let us construct symmetric doubly stochastic matrix with matrix_entries
    m = spectrum.size
    for k in range(m):
        for l in range(k, m):
            value = matrix_entries(k, l, spectrum)
            if value < 0.0:
                return False  # spectrum.size+1,value,spectrum, k
    return True

def dsm_projection(K, max_iter=1000, method='KL'):
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/PID1516079.pdf
    # Assume symmetric K with non-negative entries
    P = np.clip(K, a_min=0, a_max=None)
    assert np.allclose(P, P.T)
    n = len(K)
    for i in range(max_iter):
        prev_P = np.array(P)

        if method == 'square':
            P_row_sum = np.sum(P, axis=0, keepdims=True)
            temp = np.sum(np.eye(n) - P + (1./n) * P_row_sum, axis=1, keepdims=True)
            P = P + (1./n) * (temp - P_row_sum)
            P = np.clip(P, a_min=0, a_max=None)
        elif method == 'KL':
            P /= np.sum(P, axis=0, keepdims=True)
            P /= np.sum(P, axis=1, keepdims=True)
        else:
            raise ValueError('Unknown method')

        if np.allclose(P, prev_P):
            print(f"finished in {i} iterations")
            break

    return P

def vectorized_constr_ds_matrix(n, eig_sign='pos'):
    # n: int, size of the spectrum
    # returns: np.array, a symmetric doubly stochastic matrix, with spectrume that sums up to 0.5
    m = n - 1

    if eig_sign=='pos':
        spectrum = np.ones(m) * (0.5 / m)  # sum to 0.5
    elif eig_sign == 'neg':
        spectrum = - np.ones(m) * (0.5 / m)  # sum to -0.5
    else:
        raise ValueError('Unknown eig sign')
    spectrum = np.concatenate([[1], spectrum])

    basis = np.arange(n)[:, None]
    j_basis = np.arange(1, n)[:, None]
    sin_basis = np.sin(basis @ j_basis.T * (2 * np.pi / n) + np.pi / 4)  # for k, l
    rescaled_basis = sin_basis * spectrum[1:]
    P = (1./n) * (1 + 2 * sin_basis @ rescaled_basis.T)
    return P, spectrum

""" classification tasks """
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
def classify_error_reweighting(Y, Yhat):
    labels = np.argmax(Y, axis=1)
    # Calculate class weights to balance the classes
    # this is the same as sklearn metrics balanced accuracy score
    # unique_classes, class_counts = np.unique(labels, return_counts=True)
    # class_weights = class_counts.sum() / (len(unique_classes) * class_counts)
    # sample_weights = class_weights[labels.astype(int)]
    # Compute the testing error with class reweighting
    # weighted_error = np.mean((labels != Yhat) * sample_weights)
    weighted_error = 1. - balanced_accuracy_score(labels, Yhat)
    # print('compare scipy and weighted error is :: ', 1.-balanced_accuracy_score(labels, Yhat), weighted_error)
    return weighted_error


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
    print(' the rmse is ------------------------------------------------- ', prefix, testrmse, y_test.shape, predictions.shape)
    error_dict[prefix+'-high'].append(testrmsehigh)
    error_dict[prefix+'-low'].append(testrmselow)



''' statistics '''

''' incrementally update empirical mean after observing a mini batch of samples Y '''
def update_emp_mean(samples, empirical_mean, t, mini_batch_size):
    empirical_mean \
        = (empirical_mean * t * mini_batch_size + np.sum(samples, axis=0)) \
          / ((t + 1) * mini_batch_size)
    return empirical_mean


if __name__ == '__main__':
    x = np.random.uniform(0., 1., [3, 3])
    get_fastsimP(x)
