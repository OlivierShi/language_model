import numpy as np
import theano
from theano import tensor as T


class softmax(object):
    def __init__(self, n_input, n_output, x_r, x_c):
        self.n_input = n_input
        self.n_output = n_output
        self.logit_shape = x_r.shape
        self.x_r = x_r.reshape([self.logit_shape[0]*self.logit_shape[1], self.logit_shape[2]])
        self.x_c = x_c.reshape([self.logit_shape[0]*self.logit_shape[1], self.logit_shape[2]])

        floatX = theano.config.floatX

        init_Wr = np.asarray(np.random.uniform(low=-np.sqrt(1./self.n_input),
                                               high=np.sqrt(1./self.n_input),
                                               size=(self.n_input, self.n_output)),
                             dtype=floatX)
        init_Wc = np.asarray(np.random.uniform(low=-np.sqrt(1./self.n_input),
                                               high=np.sqrt(1./self.n_input),
                                               size=(self.n_input, self.n_output)),
                             dtype=floatX)
        init_br = np.zeros((n_output), dtype=floatX)
        init_bc = np.zeros((n_output), dtype=floatX)
        self.Wr = theano.shared(value=init_Wr, name='row_output_W')
        self.Wc = theano.shared(value=init_Wc, name='column_output_W')
        self.br = theano.shared(value=init_br, name='row_output_b')
        self.bc = theano.shared(value=init_bc, name='column_output_c')
        self.params = [self.Wr, self.Wc, self.br, self.bc]

        self.build()

    def build(self):
        self.activation_r = T.nnet.softmax(T.dot(self.x_c, self.Wr) + self.br)
        self.activation_c = T.nnet.softmax(T.dot(self.x_r, self.Wc) + self.bc)

        self.predict_r = T.argmax(self.activation_r, axis=-1)
        self.predict_c = T.argmax(self.activation_c, axis=-1)
