import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import numpy as np
from networks.imnetworks import create_image_classification_nn,\
    create_image_classification_uncertain_nn, vectorize_vars, mask_vars, get_mask
from agents.baselearner import baselearner
from scipy import stats
#from scipy import ndimage
# import cv2
import math
import random
from utils.utils import *

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

class image_classifier(baselearner):
    def __init__(self, params):
        super(image_classifier, self).__init__(params)
        self.n = 0

        self.input, self.target, self.prediction, self.advx, self.loss, self.tvars\
                = create_image_classification_nn(self.name, self.config)

        self.thres = tf.placeholder(name='thres', dtype=tf.float32, shape=())
        self.masks = None
        self.masks_op = get_mask(self.tvars, self.thres)
        self.tvarsbackup = []

        self.gradtest = tf.gradients(self.loss, self.tvars)

        self.params_update = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def log_grad(self, X, Y):
        grads = self.sess.run(self.gradtest, feed_dict={self.input: X, self.target: Y})
        for ix, grad in enumerate(grads):
            if 'grad-norm-' + str(ix) not in self.loginfo.error_dict:
                self.loginfo.error_dict['grad-norm-' + str(ix)] = []
            self.loginfo.error_dict['grad-norm-' + str(ix)].append(np.sum(np.abs(grad)))

    def compute_dist_mismatch(self, topk=100):
        datax, _ = sample_from_subset([0, 1], self.mydatabag.x_train, self.mydatabag.y_train, mini_batch_size=320)
        dataxpm, _ = sample_from_subset([2, 3], self.mydatabag.x_train, self.mydatabag.y_train, mini_batch_size=320)
        phix = self.sess.run(self.phi, feed_dict={self.input: datax})
        phixpm = self.sess.run(self.phi, feed_dict={self.input: dataxpm})
        u, s, _ = np.linalg.svd(phix, full_matrices=True)
        upm, spm, _ = np.linalg.svd(phixpm, full_matrices=True)
        avg_innerprod = np.mean(np.abs(np.sum(u[:, :topk] * upm[:, :topk], axis=0)[:topk]))
        print(' the avg inner prod is ================================== ', avg_innerprod)
        print(' the rank of the two spaces is ========================== ',
              phix.shape, s.shape, np.sum(s>1e-10), np.sum(spm>1e-10),
              np.linalg.matrix_rank(phix), np.linalg.matrix_rank(phixpm))

    def train(self, X, Y):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})
        # print(' the mini batch dimension is ------------- ', X.shape, Y.shape)
        # if self.n % 500 == 0:
        #    self.compute_dist_mismatch()
        #    self.log_grad(X, Y)
        self.n += 1

    def genadvsamples(self, X, Y):
        Xadv = self.sess.run(self.advx, feed_dict={self.input: X, self.target: Y})
        if self.n % 100 == 0:
            print(np.linalg.norm(Xadv[1] - X[1]), np.max(np.abs(Xadv[1] - X[1])))
            print(np.linalg.norm(Xadv[0] - X[0]), np.max(np.abs(Xadv[0] - X[0])))
        return Xadv

    def predict(self, X):
        # 64-by-64 attack can drop to 85% with gaussian noise 0.3; drop to 66% with uniform(0, 0.3)
        # 128-by-128 drop to 89%
        # 256^2 drop to 90%; uniform 67%
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        # X = X + np.random.normal(0., math.sqrt(0.01), X.shape)
        X = self.genadvsamples(self.mydatabag.x_test, self.mydatabag.y_test)
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        return np.squeeze(predictions)

    def mask_vars(self, perc):
        vec = self.sess.run(vectorize_vars(self.tvars))
        thres = np.quantile(np.abs(vec), perc)
        self.masks = self.sess.run(self.masks_op, {self.thres: thres})
        print(' the thres is ---------------------------------- ', perc, thres, np.sum(np.abs(vec)<thres)/vec.shape[0])
        # those masked should be zero
        self.sess.run(mask_vars(self.tvars, self.masks))
        return

    def recover_op(self):
        for idx, v in enumerate(self.tvarsbackup):
            self.sess.run(tf.assign(self.tvars[idx], v))

    def compute_error_backup(self, final_log=False):
        print(' compute acc at step ----------------------------------------- ', self.n)
        yhat = self.predict(self.mydatabag.x_test)
        testerr = classify_error(self.mydatabag.y_test, yhat, True)
        print(' the testing acc before masking is ', 1. - testerr)
        self.tvarsbackup = [self.sess.run(v) for v in self.tvars]
        for perc in [0.97, 0.95, 0.93, 0.86]:
            self.mask_vars(perc)
            yhat = self.predict(self.mydatabag.x_test)
            testerr = classify_error(self.mydatabag.y_test, yhat, True)
            print(' the testing acc after masking is ', 1. - testerr)
            self.recover_op()
        return


class RPimage_classifier(baselearner):
    def __init__(self, params):
        super(RPimage_classifier, self).__init__(params)
        self.n = 0

        self.input, self.target, self.prediction, self.phi, self.loss, self.tvars\
                = create_image_classification_nn(self.name, self.config)

        self.projdim = 500
        self.rpstd = 1./np.sqrt(self.projdim)
        # self.rpstd = 1./np.sqrt(self.projdim)
        self.n_samples = 500

        self.RPtransform(self.n_samples, self.projdim, self.rpstd)
        # self.RPtransform(self.n_samples, self.projdim, 1.)
        import matplotlib.pyplot as plt
        for _ in range(20):
            i = np.random.randint(0, self.x_train.shape[0])
            plt.imshow(self.x_train[i], cmap='gray')
            plt.show()

            plt.imshow(self.mydatabag.x_train[i], cmap='gray')
            plt.show()
        exit()
        print(np.mean(self.x_train[0]), np.median(self.x_train[0]), np.mean(self.x_train), np.median(self.x_train))

        mini_batch_inds = random.sample(range(self.x_train.shape[0]), k=32)
        self.mydatabag.x_train = self.mydatabag.x_train[mini_batch_inds]
        self.mydatabag.y_train = self.mydatabag.y_train[mini_batch_inds]

        self.mydatabag.x_train = np.vstack([self.mydatabag.x_train, self.x_train])
        self.mydatabag.y_train = np.vstack([self.mydatabag.y_train, self.y_train])

        print(' finished transformation', self.y_train.shape, self.x_train.shape,
              self.mydatabag.x_train.shape, self.mydatabag.y_train.shape)

        self.params_update = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def gen_random_unit(self, n, k):
        def get_unitvec(k):
            v = np.random.normal(0., 1., k)
            v = v / np.linalg.norm(v, ord=2)
            return v
        randomvecs = np.zeros((n, k))
        randomvecs[:k, :k] = np.eye(k)
        ind = k
        i = 0
        # thres = np.cos(math.pi/2 + 0.08 * 0.037 * math.pi/2)
        thres = 1e-5
        while ind < n:
            v = get_unitvec(k)
            if np.all(np.abs(np.mean(randomvecs[:ind, :].dot(v))) < np.abs(thres)):
                randomvecs[ind, :] = v
                ind += 1
            i += 1
            if i % 2000 == 0:
                print(ind, i, np.mean(np.abs(randomvecs[:ind, :].dot(v))))
        print(' used total loop ', i)
        return randomvecs

    def RPtransform(self, n, k, std):
        self.randommat = np.random.normal(0, std, [n, int(k)])
        # self.randommat /= np.linalg.norm(self.randommat, ord=2, axis=1).reshape((-1, 1))
        # self.randommat = self.gen_random_unit(n, k)
        # self.randommat = np.random.choice([-1/np.sqrt(k), 1/np.sqrt(k)], [n, int(k)])
        # self.randommat /= np.linalg.norm(self.randommat, ord=2, axis=1).reshape((-1, 1))
        print('the avg norm of original data is:: ',
              np.mean(np.linalg.norm(self.mydatabag.x_train.reshape((self.mydatabag.x_train.shape[0], -1)), ord=2, axis=1)))
        x_train = self.randommat.T.dot(self.mydatabag.x_train[:n].reshape((n, -1)))
        x_train = self.randommat.dot(x_train)
        print('np.linalg.norm(x_train, ord=2, axis=-1, keepdims=True)',
              np.linalg.norm(x_train, ord=2, axis=-1, keepdims=True).shape,
              np.max(x_train, axis=-1, keepdims=True).shape)
        x_train = np.clip(x_train/np.max(x_train, axis=-1, keepdims=True), 0., 1.)
        # x_train = np.clip(np.abs(x_train), 0., 1.)
        # x_train[x_train<0.0] = 0
        # x_train[x_train>0.0] = 1
        # x_train[x_train > 0.5] = 1.
        # x_train = np.clip(x_train, 0., 1.)

        print('the avg norm of projected data is:: ', np.mean(np.linalg.norm(x_train, ord=2, axis=1)),
              np.linalg.norm(self.randommat, ord=2, axis=1).shape)
        self.x_train = x_train.reshape([-1] + self.config.dim)
        self.y_train = self.mydatabag.y_train[:n]
        # self.y_train = self.randommat.dot((self.randommat.T.dot(self.y_train)))

    def postprocessing(self, x):
        kernel = np.array([[-2, -2, -2],
                           [-2, 32, -2],
                           [-2, -2, -2]])
        x = cv2.filter2D(src=x, ddepth=-1, kernel=kernel).reshape(self.config.dim)
        return x
            # self.x_train[i] = self.x_train[i]/np.max(self.x_train[i])
            # self.x_train[i] = cv2.divide(self.x_train[i], bg, scale=1).reshape(self.config.dim)
            # self.x_train[i] = ndimage.gaussian_filter(self.x_train[i], 5)

    def ClassbasedRPtransform(self, n, k, std):
        self.randommat = np.zeros([n, int(k)])
        self.randvecs = np.random.normal(0., std, [10, int(k)])
        for i in range(n):
            self.randommat[i, :] = self.randvecs[np.argmax(self.mydatabag.y_train[i]), :]
        print('the avg norm of original data is:: ',
              np.mean(np.linalg.norm(self.mydatabag.x_train.reshape((self.mydatabag.x_train.shape[0], -1)), ord=2, axis=1)))
        x_train = self.randommat.T.dot(self.mydatabag.x_train[:n].reshape((n, -1)))
        x_train = self.randommat.dot(x_train)
        # x_train = np.clip(x_train, 0., 1.)
        print('the avg norm of projected data is:: ', np.mean(np.linalg.norm(x_train, ord=2, axis=1)),
              np.linalg.norm(self.randommat, ord=2, axis=1).shape)
        self.x_train = x_train.reshape([-1] + self.config.dim)
        self.y_train = self.mydatabag.y_train[:n]
        # self.y_train = self.randommat.dot((self.randommat.T.dot(self.mydatabag.y_train)))

    def train(self, X, Y):
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})
        self.n += 1

    def predict(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        predictions = self.sess.run(self.prediction, feed_dict={self.input: X})
        return np.squeeze(predictions)

    def ensemble_predict(self, X):
        preds = []
        for i in range(50):
            self.randommat = np.random.normal(0, self.rpstd, [X.shape[0], int(self.projdim)])
            # self.randommat = np.random.normal(0, 1., [n, int(k)])
            xtest = self.randommat.T.dot(X.reshape((X.shape[0], -1)))
            xtest = self.randommat.dot(xtest).reshape([-1]+self.config.dim)
            preds.append(self.predict_help(xtest))
        preds = np.vstack(preds).T
        print(' the shape of preds is :: ', preds.shape, X.shape)
        predictions = stats.mode(preds, axis=1)[0]
        return np.squeeze(predictions)