import tensorflow as tf
import numpy as np
from networks.imnetworks import create_implicit_image_classifier_nn
from agents.baselearner import baselearner

class implicit_image_classifier(baselearner):
    def __init__(self, params):
        super(implicit_image_classifier, self).__init__(params)
        self.n = 0

        self.input_layer_x, self.target_input, self.implicit_loss, self.lossvec, self.grad_holders, self.tvars \
                = create_implicit_image_classifier_nn(scopename=self.name, config=self.config)

        self.total_loss = self.implicit_loss

        #self.grads = tf.gradients(self.total_loss, self.tvars)
        self.opt = tf.train.AdamOptimizer(self.config.learning_rate)
        #self.params_update = self.opt.apply_gradients(zip(self.grad_holders, self.tvars))
        self.params_update_batch = self.opt.minimize(self.total_loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # self.train = self.train_manual_agg

    ''' this training method is incorrect '''
    def train(self, X, Y):
        if len(X.shape) < 2:
            X = X.reshape((-1, self.config.dim))
        self.sess.run(self.params_update_batch, feed_dict={self.input_layer_x: X, # + np.random.normal(0., 0.2, X.shape),
                                                        self.target_input: np.argmax(Y,axis=1)[:,None]})

    def predict(self, X):
        if len(X.shape) < len(self.config.dim):
            X = X.reshape((X.shape[0], self.config.dim))

        predictions = np.zeros(X.shape[0])
        Y = np.arange(10).reshape((-1,1))

        ys = 10
        # Y = np.linspace(0., 10.5, ys).reshape([-1, 1])+np.random.normal(0., 0.1, xx[None,:].shape)

        for i in range(X.shape[0]):
            xx = X[i, :] + np.random.normal(0., 0.2, X[i, :].shape)
            xx = np.repeat(xx[None,:], [ys], axis=0)
            #print(xx.shape)
            lossvec = self.sess.run(self.lossvec, feed_dict={self.input_layer_x: xx, self.target_input: Y})
            #print(lossvec.shape)
            predictions[i] = round(Y[np.argmin(lossvec),0])

        print(predictions.shape, lossvec.shape, lossvec)
        return predictions
