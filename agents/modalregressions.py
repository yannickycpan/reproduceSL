import tensorflow as tf
import numpy as np
from networks.networks import create_implicit_nn, create_mdn_nn
from agents.baselearner import baselearner
from utils.utils import get_global_prediction, get_local_prediction, list_to_numpy_mat, modal_prediction_yhat_log


def predict(config, sess, lossnode, inputXnode, inputYnode, X):
    if len(X.shape) < 2:
        X = X.reshape((X.shape[0], config.dim))
    predictions = np.zeros(X.shape[0])
    discretization = config.discretization
    Y = np.linspace(config.low, config.high, discretization)
    Y = Y.reshape((-1, 1)) if len(inputYnode.shape) > 1 else Y
    for i in range(X.shape[0]):
        xx = X[i, :].reshape((-1, config.dim))
        xx = np.repeat(xx, [discretization], axis=0)
        lossvec = sess.run(lossnode, feed_dict={inputXnode: xx, inputYnode: Y})
        predictions[i] = Y[np.argmin(lossvec)]
    return predictions


def get_all_modes(config, sess, lossnode, inputXnode, inputYnode, X):
    if len(X.shape) < 2:
        X = X.reshape((X.shape[0], config.dim))
    ''' this discretization is used for finite difference method '''
    discretization = config.discretization
    predictions_local = []
    avgmodes_local = []
    predictions_global = []
    avgmodes_global = []
    Y = np.linspace(config.low, config.high, discretization)
    Y = Y.reshape((-1, 1)) if len(inputYnode.shape) > 1 else Y
    for i in range(X.shape[0]):
        xx = X[i, :].reshape((-1, config.dim))
        xx = np.repeat(xx, [discretization], axis=0)
        lossvec = sess.run(lossnode, feed_dict={inputXnode: xx, inputYnode: Y})

        pred_local, nummodes_local = get_local_prediction(lossvec, np.squeeze(Y))
        pred_global, nummodes_global = get_global_prediction(lossvec, np.squeeze(Y), config.epsilon)

        avgmodes_local.append(nummodes_local)
        predictions_local.append(pred_local)
        avgmodes_global.append(nummodes_global)
        predictions_global.append(pred_global)
    predictions_local = list_to_numpy_mat(predictions_local, maskval=np.nan)
    predictions_global = list_to_numpy_mat(predictions_global, maskval=np.nan)
    return predictions_local, np.mean(avgmodes_local), predictions_global, np.mean(avgmodes_global)


class nnlearner_implicit(baselearner):
    def __init__(self, params):
        super(nnlearner_implicit, self).__init__(params)
        print('param algorithms are :: ', params)
        self.input, self.target, self.loss, self.lossvec, \
        self.fxy, self.hidden2, self.xreps, self.der_loss, self.testW, self.grad_holders, self.tvars \
                = create_implicit_nn(self.name, self.config)

        self.opt = tf.train.AdamOptimizer(self.config.learning_rate)
        self.params_update = self.opt.minimize(self.loss)

        self.grads = tf.gradients(self.loss, self.tvars)
        #self.firstloss_grad = tf.gradients(tf.reduce_mean(self.firstlossvec), self.tvars)
        self.params_update_manul = self.opt.apply_gradients(zip(self.grad_holders, self.tvars))

        # for debug
        self.grads_wrt_w = tf.gradients(self.der_loss, self.tvars)[0]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        """ use a customized logger function """
        self.compute_error = self.compute_error_modal

        self.n = 0

    # this is fine on small domain
    def train(self, X, Y):
        if len(X.shape) < 2:
            X = X.reshape((X.shape[0], self.config.dim))
        if len(Y.shape) < 2:
            Y = Y.reshape((X.shape[0], 1))
        #if self.n % 2000 == 0 and self.n > 10000:
        #    modal_prediction_yhat_log(self.mydatabag, self.name + '_' + str(self.n), self.get_all_modes)
        #    self.compute_loss()
            # self.compute_grad2target()
        #    grad = self.sess.run(self.grad2target, feed_dict={self.input: X, self.target: Y})
        #    print(grad)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})
        self.n += 1
        #if self.n % 100000 == 0:
        #    testgrad = self.sess.run(self.grads_wrt_w, feed_dict={self.input: X, self.target: Y})
        #    print('grad magnitude test -------------- ', np.sum(np.abs(testgrad)))

    def aggregate_mini_batch_gradient(self, X, Y):
        ''' X[[i], ]: slice and keep dim '''
        sumgrads = self.sess.run(self.grads, feed_dict={self.input: X[[0], ], self.target: Y[[0], ]})
        for i in range(1, X.shape[0]):
            grads = self.sess.run(self.grads, feed_dict={self.input: X[[i], ], self.target: Y[[i], ]})
            for id, grad in enumerate(grads):
                sumgrads[id] = sumgrads[id] + grad
        avggrads = [grad / float(X.shape[0]) for grad in sumgrads]
        return avggrads

    def train_manual_aggregate(self, X, Y):
        if len(X.shape) < 2:
            X = X.reshape((X.shape[0], self.config.dim))
        if len(Y.shape) < 2:
            Y = Y.reshape((X.shape[0], 1))
        #if self.n % 2000 == 0:
        #    self.compute_loss()
        self.n += 1
        avggrads = self.aggregate_mini_batch_gradient(X, Y)
        self.sess.run(self.params_update_manul, feed_dict=dict(zip(self.grad_holders, avggrads)))

    def predict(self, X):
        predictions = predict(self.config, self.sess, self.lossvec, self.input, self.target, X)
        return predictions

    def get_all_modes(self, X):
        predictions_local, avgmodes_local, predictions_global, avgmodes_global \
            = get_all_modes(self.config, self.sess, self.lossvec, self.input, self.target, X)
        return predictions_local, avgmodes_local, predictions_global, avgmodes_global

    def compute_loss(self, dataset = None):
        discretization = int(2 * self.config.discretization)
        X = np.linspace(self.config.low-1., self.config.high + 1., discretization).reshape([-1, self.config.dim])
        Y = np.linspace(self.config.low-1., self.config.high + 1., discretization).reshape([-1, 1])
        errors = []
        totalerrors = []
        for i in range(X.shape[0]):
            xx = X[i, :].reshape((-1, self.config.dim))
            xx = np.repeat(xx, [discretization], axis=0)
            fxy, lossvec = self.sess.run([self.fxy, self.lossvec], feed_dict={self.input: xx, self.target: Y})
            totalerrors.append(np.reshape(lossvec, [-1, discretization]))
            errors.append(np.reshape(fxy, [-1, discretization]))
        errors = np.vstack(errors)
        totalerrors = np.vstack(totalerrors)
        np.savetxt(self.name + 'fxy' + str(self.n) + '.txt', errors, fmt='%10.7f', delimiter=',')
        np.savetxt(self.name + 'totalloss' + str(self.n) + '.txt', totalerrors, fmt='%10.7f', delimiter=',')
        return errors

    def get_representation(self, X):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        Y = self.predict(X)
        Y = Y.reshape((-1, 1))
        repres = self.sess.run(self.hidden2, feed_dict={self.input: X, self.target: Y})
        return repres


class mdnlearner(baselearner):
    def __init__(self, params):
        super(mdnlearner, self).__init__(params)

        self.input, self.target, self.component_w, self.mus,\
        self.loss, self.neg_log_prob, _ \
                = create_mdn_nn(self.name, self.config)
        self.params_update = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.compute_error = self.compute_error_modal

    def train(self, X, Y):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        if len(Y.shape) > 1:
            Y = Y.reshape(-1)
        self.sess.run(self.params_update, feed_dict={self.input: X, self.target: Y})

    def predict(self, X):
        predictions = predict(self.config, self.sess, self.neg_log_prob, self.input, self.target, X)
        return predictions

    def get_all_modes(self, X):
        predictions_local, avgmodes_local, predictions_global, avgmodes_global \
            = get_all_modes(self.config, self.sess, self.neg_log_prob, self.input, self.target, X)
        return predictions_local, avgmodes_local, predictions_global, avgmodes_global