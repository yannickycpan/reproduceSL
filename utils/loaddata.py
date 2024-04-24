import numpy as np
#import os
import math
#import sys

class databag:

    def __init__(self, name):
        self.minval = None
        self.maxval = None
        self.width = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_valid = None
        self.y_valid = None

        self.y_test_true_modes = None
        self.label_to_samples = []
        self.name = name
        self.covy = None
        self.dim = None
        self.n_classes = None
        self.wholedataset = None
        if name == 'sin_hc':
            self.query_y = self.query_y_sin_hc
        else:
            self.query_y = None
        ''' this is for biased circle data only '''
        self.most_likely_mode = None

    def biased_circle_most_likely(self, x):
        truevals = np.zeros(x.shape[0])
        x = np.squeeze(x)
        truevals[x <= 0.] = np.sqrt(1 - np.square(x[x <= 0]))
        truevals[x > 0.] = -1. * np.sqrt(1 - np.square(x[x > 0]))
        self.most_likely_mode = truevals

    def query_y_sin_hc(self, x):
        if x < 0:
            y = np.sin(8 * math.pi * x)
        else:
            y = np.sin(0.5 * math.pi * x)
        return y

    def init_transform(self, target):
        self.minval = np.min(target)
        self.maxval = np.max(target)
        self.width = self.maxval - self.minval

    def transform(self, target):
        return (target - self.minval)/self.width

    def inv_transform(self, target):
        if self.width is None or self.minval is None:
            return target
        return target*self.width + self.minval

    def set_data_groups(self, x_train, y_train, x_test, y_test, x_valid=None, y_valid=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_valid = x_valid
        self.y_valid = y_valid
        #self.dim = [1]
        #if len(y_train.shape) > 1:
        #    self.covy = 1./float(y_train.shape[0]) * y_train.T.dot(y_train)
        #    self.nclasses = y_train.shape[1]
        if len(x_train.shape) > 2:
            self.dim = list(x_train.shape[1:])
            print(' the shape is -------------------- ', self.dim)
        elif len(x_train.shape) <= 2:
            self.dim = x_train.shape[1]
        #self.n_classes = self.y_test.shape[1]

    ''' this is applicable for multitargets data sets '''
    def compute_true_modes(self):
        x_test, y_test = self.x_test, self.y_test
        datasetname = self.name
        if len(x_test.shape) < 2:
            x_test = x_test.reshape((-1, 1))
        if len(y_test.shape) < 2:
            y_test = y_test.reshape((-1, 1))
        truevals = y_test.copy()
        if datasetname == "circle":
            truevals = np.hstack([-y_test, y_test])
        elif datasetname in ["double_circle"]:
            ''' there will be nan values, but it does not affect error calculation '''
            y1 = np.sqrt(1. - np.square(x_test))
            y2 = -y1
            y3 = np.sqrt(4. - np.square(x_test))
            y4 = -y3
            truevals = np.hstack([y1, y2, y3, y4])
            #print(x_test[:10, :])
            #print(truevals[:10, :])
        elif datasetname == "high_double_circle":
            y1 = np.sqrt(1. - np.square(y_test))
            y2 = -y1
            y3 = np.sqrt(4. - np.square(y_test))
            y4 = -y3
            truevals = np.hstack([y1, y2, y3, y4])
        elif datasetname == "biased_circle":
            self.biased_circle_most_likely(x_test)
        self.y_test_true_modes = truevals

    def get_subset_label(self, y_train):
        if self.n_classes is None or self.n_classes < 2:
            return
        self.label_to_samples = []
        indexes = np.arange(y_train.shape[0])
        for i in range(self.n_classes):
            subset = indexes[np.argmax(y_train, axis=1) == i]
            self.label_to_samples.append(subset)


def split2trainvalid(dataset, trainsize=None, targetdim = 1):
    if trainsize is None:
        trainsize = int(0.8 * dataset.shape[0])
    x_train, y_train = dataset[:trainsize, :-targetdim], dataset[:trainsize, -targetdim:]
    x_test, y_test = dataset[trainsize:, :-targetdim], dataset[trainsize:, -targetdim:]
    return x_train, np.squeeze(y_train), x_test, np.squeeze(y_test)


def splitdata(dataset, prop=0.9):
    return dataset[:int(prop*dataset.shape[0])], dataset[int(prop*dataset.shape[0]):]


def convert2onehot(nparr):
    print('number of categories is --------------- ', len(np.unique(nparr)))
    onehoty = np.zeros((len(nparr), len(np.unique(nparr))))
    onehoty[np.arange(len(nparr)), nparr.astype(int)] = 1.0
    return onehoty


def preprocess2lenet(x_train, x_test, pad=True):
    x_train /= 255.
    x_test /= 255.
    print(' the shape of x train is ::: ', x_train.shape)
    if pad:
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    return x_train, x_test


def balance_dataset(X, Y):
    # Find indices of positive and negative samples
    positive_indices = np.where(Y == 1)[0]
    negative_indices = np.where(Y == 0)[0]

    # Randomly sample from the negative class to match the size of the positive class
    num_positive_samples = len(positive_indices)
    balanced_indices = np.random.choice(negative_indices, size=num_positive_samples, replace=False)

    # Combine indices of positive and negative samples
    balanced_indices = np.concatenate((positive_indices, balanced_indices))

    # Shuffle the indices to ensure a random order
    np.random.shuffle(balanced_indices)

    # Use the balanced indices to select samples from the dataset
    balanced_X = X[balanced_indices]
    balanced_Y = Y[balanced_indices]

    return balanced_X, balanced_Y


def repeat_data_points(X, n):
    original_num_points = X.shape[0]

    var = np.var(X, axis=0)/X.shape[0]

    # Calculate how many times each data point needs to be repeated
    repetitions_per_point = n // original_num_points

    # Calculate the remainder for additional repetitions
    remainder = n % original_num_points

    # Repeat each data point the required number of times
    repeated_data = np.repeat(X, repetitions_per_point, axis=0)

    # Randomly select additional data points to repeat for the remainder
    if remainder > 0:
        additional_indices = np.random.choice(original_num_points, remainder, replace=False)
        additional_data = X[additional_indices] + np.random.normal(0., np.sqrt(var), X[additional_indices].shape)
        repeated_data = np.vstack((repeated_data, additional_data))

    # Shuffle the repeated data to ensure randomness
    np.random.shuffle(repeated_data)

    return repeated_data

def get_data(datasetname, datadir, addnoise, subsample):
    mydatabag = databag(datasetname)
    datafile = datadir + '/' +datasetname
    x_train = y_train = x_test = y_test = x_valid = y_valid = None
    if datasetname in ["toy_dataset", "toy_dataset_add2x"]:
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        # dataset = dataset[:20000,:]
        np.random.shuffle(dataset)
        x_train, y_train, x_test, y_test = split2trainvalid(dataset)
        if addnoise == 1:
            y_train = y_train + np.random.normal(0., 0.1, len(y_train))
        elif addnoise == 2:
            y_train[x_train < 0] = y_train[x_train < 0] + np.random.normal(0., 1., len(y_train[x_train < 0]))
        elif addnoise == 3:
            y_train[x_train > 0] = y_train[x_train > 0] + np.random.normal(0., 1., len(y_train[x_train > 0]))
    elif datasetname in ["sin_hc"]:
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        np.random.shuffle(dataset)
        dataset = dataset[:min(dataset.shape[0], 5000), :]
        if subsample < 1.:
            dataset = dataset[np.random.randint(0, dataset.shape[0], int(subsample * dataset.shape[0])), :]
        x_train, y_train, x_test, y_test = split2trainvalid(dataset)
        y_train = y_train + np.random.normal(0., addnoise, len(y_train))
    elif datasetname in ["circle", "double_circle", "inverse_sin", "biased_circle"]:
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        np.random.shuffle(dataset)
        dataset = dataset[:min(dataset.shape[0], 5000), :]
        x_train, y_train, x_test, y_test = split2trainvalid(dataset)
        if datasetname == "inverse_sin":
            y_train = y_train + np.random.uniform(-0.1, 0.1, len(y_train))
        elif addnoise == 1:
            y_train = y_train + np.random.normal(0., 0.1, len(y_train))
        if datasetname == "biased_circle":
            truevals = np.zeros_like(y_test)
            truevals[x_test[:, 0] <= 0.] = np.sqrt(1 - np.square(x_test[x_test[:, 0] <= 0., 0]))
            truevals[x_test[:, 0] > 0.] = -1. * np.sqrt(1 - np.square(x_test[x_test[:, 0] > 0., 0]))
            y_test_high = truevals.reshape((-1, 1))
            y_test_low = -truevals.reshape((-1, 1))
            y_test = np.hstack([y_test_high, y_test_low])
    elif datasetname in ["high_double_circle"]:
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        x_train, y_train, x_test, y_test = split2trainvalid(dataset)
        testset = np.loadtxt(datadir + '/' +datasetname + "_test.txt", delimiter=',')
        x_test = testset[:, :-1]
        y_test = testset[:, -1]
        if addnoise == 1:
            y_train = y_train + np.random.normal(0., 0.1, len(y_train))
    elif datasetname == "calhouse":
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        from sklearn.model_selection import train_test_split
        X = housing.data
        y = housing.target
        X = X[:1000]
        y = y[:1000]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    elif datasetname in ["cpposdata", "subcpposdata", "songyeardata", "readyhousedata", "realestate","sgemmdata", "3mm_exec_times",
                         "bikesharedata", "toy_yaoli", "insurance", "exec_times", "humanposdata"]:
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        #if datasetname == "songyeardata" and addnoise == 0:
        #    trainsize = 463715
        np.random.shuffle(dataset)
        if dataset.shape[0] > 1000:
            dataset = dataset[:1000]
        trainsize = int(0.6*dataset.shape[0])
        x_train, y_train, x_test, y_test = split2trainvalid(dataset, trainsize)
    elif datasetname == 'mpg':
        data = np.load('auto-mpg.npz')
        x_train, y_train, x_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    elif datasetname in ["mcartestdata", "mcardata1k", "sin2pi"]:
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        np.random.shuffle(dataset)
        print(dataset.shape)
        trainsize = int(0.6 * dataset.shape[0])
        x_train, y_train, x_test, y_test = split2trainvalid(dataset, trainsize)
    elif datasetname == 'insurancefirsttrain':
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        mydatabag.wholedataset = dataset.copy()
        x_train, y_train = dataset[:, :-1], dataset[:, -1:]
        x_test, y_test = x_train.copy(), y_train.copy()
    elif datasetname == 'insurancemodal':
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        np.random.shuffle(dataset)
        x_train, y_train, x_test, y_test = split2trainvalid(dataset, None, 2)
        mydatabag.init_transform(y_train)
        y_train = mydatabag.transform(y_train)
        y_train = y_train[:, -1:]
    elif datasetname in ['heart', "ausweather", "travelinsurance", "cancer", "ionosphere"]:
        ''' travelinsurance is highly imbalanced '''
        mydatabag.n_classes = 2
        dataset = np.loadtxt(datafile + ".txt", delimiter=',')
        np.random.shuffle(dataset)
        x_train, y_train, x_test, y_test = split2trainvalid(dataset)
        y_train = convert2onehot(y_train)
        y_test = convert2onehot(y_test)
    elif datasetname in ['mnist', 'mnistfashion', 'cifar10', 'cifar100']:
        x_train, y_train, x_test, y_test = load_image_data(datasetname)
        x_train, y_train = shuffle(x_train, y_train)
        mydatabag.n_classes = y_test.shape[1]
    mydatabag.set_data_groups(x_train, y_train, x_test, y_test, x_valid, y_valid)
    # if 'circle' in datasetname or datasetname == 'insurancemodal':
    mydatabag.compute_true_modes()
    return mydatabag


def shuffle(x_train, y_train):
    inds = np.arange(x_train.shape[0])
    np.random.shuffle(inds)
    x_train = x_train[inds]
    y_train = y_train[inds]
    return x_train, y_train


def load_image_data(dataname):
    if dataname == 'cifar10':
        from tensorflow.keras import datasets
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    elif dataname == 'cifar100':
        from tensorflow.keras import datasets
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    elif dataname == 'mnistfashion':
        x_train, y_train, x_test, y_test = load_mnist_fashion_data()
    else:
        x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train = shuffle(x_train, y_train)
    y_train = convert2onehot(np.squeeze(y_train))
    y_test = convert2onehot(np.squeeze(y_test))
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = preprocess2lenet(x_train, x_test, False)
    return x_train, y_train, x_test, y_test


def load_mnist_data(path='mnist.npz'):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        return x_train, y_train, x_test, y_test


def load_mnist_fashion_data():
    import gzip
    #dirname = os.path.join('datasets', 'fashion-mnist')
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    #paths = []
    paths = ['mnistfashion/' + file for file in files]
    #for fname in files:
    #    paths.append(get_file(fname, origin=base + fname, cache_subdir=dirname))
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x  = np.random.uniform(0,1,[3, 5])
    print(x)
    print(repeat_data_points(x, 5))