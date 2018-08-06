import numpy as np
import theano
from theano import tensor as T
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lstm import RNN
from softmax import softmax
from updates import *


class RNNLM(object):
    def __init__(self, n_input, n_hidden, n_output, cluster_num, optimizer=sgd, p=0.5):
        self.x = T.itensor3('batched_sequence_x') # (n_maxlen, n_batch, 2)
        self.x_mask_r = T.matrix('x_mask_r')
        self.x_mask_c = T.matrix('x_mask_c')
        self.y = T.itensor3('batched_sequence_y') # (n_maxlen, n_batch, 2)
        self.y_mask = T.matrix('y_mask')

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.floatX = theano.config.floatX
        self.cluster_num = cluster_num
        self.in_cluster_num = int(np.ceil(float(self.n_output) / self.cluster_num))
        self.p = p
        init_Er = np.asarray(np.random.uniform(low=-np.sqrt(1./self.n_output),
                                               high=np.sqrt(1./self.n_output),
                                               size=(self.cluster_num, self.n_input)),
                             dtype=self.floatX)
        self.Er = theano.shared(value=init_Er, name='row_word_embedding', borrow=True)
        init_Ec = np.asarray(np.random.uniform(low=-np.sqrt(1./self.n_output),
                                               high=np.sqrt(1./self.n_output),
                                               size=(self.in_cluster_num, self.n_input)),
                             dtype=self.floatX)
        self.Ec = theano.shared(value=init_Ec, name='column_word_embedding', borrow=True)

        self.optimizer = optimizer
        self.is_train = T.iscalar('is_train')
        self.n_batch = T.iscalar('n_batch')
        self.epsilon = 1.0e-15
        self.rng = RandomStreams(1234)
        self.build()

    def build(self):
        print 'building rnn cell...'
        hidden_layer = RNN(self.rng,
                           self.n_input,self.n_hidden,self.n_batch,
                           self.x,self.Er,self.Ec,self.x_mask_r, self.x_mask_c,
                           is_train=self.is_train, p=self.p)
        print 'building softmax output layer...'
        [h_r, h_c] = hidden_layer.activation

        output_layer = softmax(self.n_hidden, self.cluster_num, self.in_cluster_num, h_r, h_c)
        cost_r = self.categorical_crossentropy(output_layer.activation_r, self.y[:,:,0])
        cost_c = self.categorical_crossentropy(output_layer.activation_c, self.y[:,:,1])
        cost = cost_r + cost_c
        self.params = [self.Er, self.Ec,]
        self.params += hidden_layer.params
        self.params += output_layer.params

        lr = T.scalar('lr')
        gparams = [T.clip(T.grad(cost, p), -10, 10) for p in self.params]
        updates = self.optimizer(self.params, gparams, lr)

        self.train = theano.function(inputs=[self.x, self.x_mask_r, self.x_mask_c, self.y, self.y_mask,
                                             self.n_batch, lr],
                                     outputs=[cost],
                                     updates=updates,
                                     givens={self.is_train: np.cast['int32'](1)})

        self.getNLL = theano.function(inputs=[self.x, self.x_mask_r, self.x_mask_c, self.n_batch],
                                      outputs=[output_layer.activation_r, output_layer.activation_c],
                                      givens={self.is_train: np.cast['int32'](0)})

        self.predict = theano.function(inputs=[self.x, self.x_mask_r, self.x_mask_c, self.n_batch],
                                       outputs=[output_layer.predict_r, output_layer.predict_c],
                                       givens={self.is_train: np.cast['int32'](0)})

        self.test = theano.function(inputs=[self.x, self.x_mask_r, self.x_mask_c, self.y, self.y_mask, self.n_batch],
                                    outputs=cost,
                                    givens={self.is_train: np.cast['int32'](0)})

    def categorical_crossentropy(self, y_pred, y_true):
        y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        y_true = y_true.flatten()
        nll = T.nnet.categorical_crossentropy(y_pred, y_true)
        return T.sum(nll * self.y_mask.flatten())


