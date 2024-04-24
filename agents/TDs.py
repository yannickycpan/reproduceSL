#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
from agents.baselearner import baselearner
from utils.utils import classify_error, sample_from_P, getP, compute_A_b_mspbe, compute_error4tds, get_seq_noise
import random
from utils.utils import acc, backfromlinear, convert2linear, introduce_noisy_labels, get_cond_noise

class tdglm(baselearner):
    def __init__(self, params):
        super(tdglm, self).__init__(params)
        ''' multitarget/label prediction '''
        self.wd = 1 if self.config.n_classes is None else self.config.n_classes
        u = np.sqrt(6./(self.wd + int(np.prod(self.config.dim)+1)))

        print('the bound is ', u)
        self.W = np.random.uniform(-u, u, (int(np.prod(self.config.dim)+1), self.wd))
        self.E = np.zeros((int(np.prod(self.config.dim)+1), 1))
        ''' add a bias unit '''
        self.x_train = self.mydatabag.x_train.reshape((-1, int(np.prod(self.config.dim))))
        self.x_train = np.hstack([self.x_train, np.ones([self.x_train.shape[0], 1])])
        self.y_train = self.mydatabag.y_train if self.config.n_classes is None \
            else introduce_noisy_labels(self.mydatabag.y_train, self.config.noiselevel)

        self.x_test = self.mydatabag.x_test.reshape((-1, int(np.prod(self.config.dim))))
        self.x_test = np.hstack([self.x_test, np.ones([self.x_test.shape[0], 1])])
        self.y_test = self.mydatabag.y_test

        print("TD algorithm is running !!!!!!!!!!! ", self.config.dim, self.config.lam, self.config.gamma, self.config.learning_rate)

        self.empirical_mean = 0.
        self.t = 0
        self.tp = None
        self.curnoise = np.random.normal(0., 1., self.config.mini_batch_size) * self.config.noiselevel
        self.nextnoise = None
        self.n = 0
        ''' if P is none, uniform const will be used by default '''
        self.P = getP(self.config.Ptype, self.mydatabag.name, self.mydatabag.x_train, self.mydatabag.y_train, self.config.epsilon)
        # print(" the matrix P is ------------------- ", self.P, self.x_train.shape)
        if self.config.n_classes is None:
            from sklearn.metrics import mean_squared_error
            n_samples = 5000
            x = self.x_train[:min(self.x_train.shape[0], n_samples)]
            y = self.y_train[:min(self.y_train.shape[0], n_samples)]

            nl, rho = 2.0, 0.8
            # noise = get_seq_noise(nl, rho, len(y) + len(self.y_test))
            totaln = len(y)+len(self.y_test)
            kmat = np.random.uniform(0., 1., [totaln, totaln])
            cov = kmat.T.dot(kmat)
            noise = np.random.multivariate_normal(np.zeros(totaln), cov)
            print(' the noise is ----------------------------- ', noise.shape, y.shape, self.config.gamma)
            y = y + noise[:len(y)]
            self.y_test = self.y_test + noise[len(y):]

            self.lrw = np.linalg.lstsq(x, y, rcond=None)[0]

            A, b, _ = compute_A_b_mspbe(x, y, np.ones((x.shape[0], x.shape[0]))*1/x.shape[0], self.config.gamma,
                                   self.config.lam)
            wtd = np.linalg.lstsq(A, b, rcond=None)[0]

            # print(getP('UnifConst', self.x_train))
            print('uniconst P dist to lr w ------------- ', np.linalg.norm(wtd - self.lrw))
            print('the acc of lr is -------------- ', mean_squared_error(self.y_test, self.x_test.dot(self.lrw.T)))
            print(' the acc of td is ------------- ', mean_squared_error(self.y_test, self.x_test.dot(wtd.T)))
            exit(0)
        self.lrw = None
        # self.closedform_solve()

    def compute_error(self, final_log=False):
        compute_error4tds(self.loginfo, self.x_train, self.y_train, self.x_test, self.y_test, self.W, self.lrw, 'TD', self.mydatabag.name)
        #compute_error4tds(self.loginfo, self.x_train, self.y_train, self.x_test, self.y_test, self.Ab_W, self.lrw,
         #                 'b-Aw')

    def sample_tp(self):
        if self.config.n_classes is None or self.config.Ptype == 'DSMXsim' or self.config.Ptype == 'FastP':
            return sample_from_P(self.P, self.x_train.shape[0], self.t)
        ylabel = sample_from_P(self.P, self.x_train.shape[0], np.argmax(self.y_train[self.t]))
        id = np.random.choice(a=self.mydatabag.label_to_samples[ylabel[0]], size=self.config.mini_batch_size)
        return id

    def train_TD(self):
        if self.n == 0:
            # self.t = random.sample(range(self.x_train.shape[0]), k=self.config.mini_batch_size)
            self.t = 0
        self.tp = self.sample_tp()

        # noisearr = get_cor_noise(self.mydatabag.name, self.config.noiselevel)
        self.nextnoise \
            = get_cond_noise(self.mydatabag.name, self.curnoise, self.config.noiselevel, self.config.mini_batch_size)

        x_t = self.x_train[self.t, :]
        y_t = self.y_train[self.t] + self.curnoise

        x_tp = self.x_train[self.tp, :]
        y_tp = self.y_train[self.tp] + self.nextnoise

        predxt = x_t.dot(self.W)
        predxtp = x_tp.dot(self.W)

        # self.empirical_mean = (self.empirical_mean * self.n + y_tp)/(self.n+1.)
        reward = convert2linear(self.mydatabag.name, y_t) - self.config.gamma * convert2linear(self.mydatabag.name, y_tp)
        tdtarget = reward.reshape(predxtp.shape) + self.config.gamma * predxtp

        if self.config.deltatype == 0:
            delta_t = backfromlinear(self.mydatabag.name, tdtarget) - backfromlinear(self.mydatabag.name, predxt)
        else:
            delta_t = tdtarget - predxt

        if self.config.mini_batch_size > 1:
            self.E = x_t.T
        else:
            self.E = self.config.gamma * self.config.lam * self.E + x_t.reshape(self.E.shape)
            self.E[-1, 0] = 1.
        deltaW = delta_t.reshape((self.wd, self.config.mini_batch_size)).dot(self.E.T)
        ''' adding 1/b is to match sgd when gamma = 0'''
        self.W = self.W + self.config.learning_rate * deltaW.T * 1./self.config.mini_batch_size

        self.t = self.tp
        self.curnoise = self.nextnoise
        self.n += 1

    def train_Ab(self):
        self.Ab_W = self.Ab_W + self.config.learning_rate * (self.b - self.A.dot(self.Ab_W))

    def train(self, X, Y):
        self.train_TD()
        # self.train_Ab()

    def closedform_solve(self):
        A, b, _ = compute_A_b_mspbe(self.x_train, self.y_train, getP('UnifConst', self.x_train), self.config.gamma,
                                    self.config.lam)
        w1 = np.linalg.lstsq(A, b, rcond=1e-10)[0]
        # print(getP('UnifConst', self.x_train))
        print('uniconst P dist to lr w ------------- ', np.linalg.norm(w1 - self.lrw),
              acc(self.x_test.dot(w1), self.y_test))

        A, b, _ = compute_A_b_mspbe(self.x_train, self.y_train, getP('DSMGram', self.x_train), self.config.gamma,
                                    self.config.lam)
        w2 = np.linalg.lstsq(A, b, rcond=1e-10)[0]
        print(' dsmgram dist to lr w ------------- ', np.linalg.norm(w2 - self.lrw),
              acc(self.x_test.dot(w2), self.y_test))

        A, b, _ = compute_A_b_mspbe(self.x_train, self.y_train, getP('NGram', self.x_train), self.config.gamma,
                                    self.config.lam)
        w3 = np.linalg.lstsq(A, b, rcond=1e-10)[0]
        print(' ngram dist to lr w ------------- ', np.linalg.norm(w3 - self.lrw),
              acc(self.x_test.dot(w3), self.y_test))

        A, b, _ = compute_A_b_mspbe(self.x_train, self.y_train, getP('Gram', self.x_train), self.config.gamma,
                                    self.config.lam)
        w5 = np.linalg.lstsq(A, b, rcond=1e-10)[0]
        print(' gram dist to lr w ------------- ', np.linalg.norm(w5 - self.lrw),
              acc(self.x_test.dot(w5), self.y_test))

        A, b, _ = compute_A_b_mspbe(self.x_train, self.y_train, getP('UnifRand', self.x_train), self.config.gamma,
                                    self.config.lam)
        w4 = np.linalg.lstsq(A, b, rcond=1e-10)[0]
        print(' unifrand dist to lr w ----------- ', np.linalg.norm(w4 - self.lrw),
              acc(self.x_test.dot(w4), self.y_test))

        print(' linear regression test loss :: ', acc(self.x_test.dot(self.lrw), self.y_test))


    ''' for predicting integers '''
    def predict(self, X):
        Xones = np.hstack([X, np.ones([X.shape[0], 1])])
        predictions = Xones.dot(self.W)
        print(" the weight norm is ----------------- ", np.linalg.norm(self.W, ord=2))
        return np.squeeze(predictions)


def compute_error(loginfo, predict, mydatabag):
    loginfo.log_error_discrete(predict, mydatabag)
    count = 0
    for i in range(0, 10, 2):
        X, Y = sample_from_subset([i, i + 1],
                                  mydatabag.x_test, mydatabag.y_test)
        if 'task' + str(count) + 'test' not in loginfo.error_dict:
            loginfo.error_dict['task' + str(count) + 'test'] = []
        else:
            predictions = predict(X)
            err = classify_error(Y, predictions, True)
            loginfo.error_dict['task' + str(count) + 'test'].append([err])
            # print(' task ------------------- ', count, ' ----------------- error is :: ', err)
        count += 1


def sample_from_subset(labels, dataX, dataY, mini_batch_size=None):
    l1, l2 = labels
    intlabels = np.argmax(dataY, axis=1)
    subset = (intlabels == l1) + (intlabels == l2)
    subX, subY = dataX[subset, :],\
                 dataY[subset, :]
    if mini_batch_size is None:
        return subX, subY
    # print(dataX.shape, subY.shape, subX.shape, mini_batch_size, labels)
    mini_batch_inds = random.sample(range(subX.shape[0]), k=mini_batch_size)
    return subX[mini_batch_inds], subY[mini_batch_inds]