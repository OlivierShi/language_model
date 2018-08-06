import numpy as np
import theano
from theano import tensor as T

class RNN:
    def __init__(self, rng,
                 n_input, n_hidden, n_batch,
                 x, Exr, Exc, mask_r, mask_c, is_train=1, p=0.5):
        self.rng = rng
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_batch = n_batch
        self.x = x
        self.Exr = Exr
        self.Exc = Exc
        self.mask_r = mask_r
        self.mask_c = mask_c
        self.f = T.nnet.sigmoid
        self.is_train = is_train
        self.p = p
        floatX = theano.config.floatX
        # parameters for row propagation
        init_Wf = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                               high=np.sqrt(1./n_input),
                                               size=(self.n_input+self.n_hidden, self.n_hidden)),
                             dtype=floatX)
        init_bf = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wf_r = theano.shared(value=init_Wf, name='Wf')
        self.bf_r = theano.shared(value=init_bf, name='bf')
        init_Wi = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                               high=np.sqrt(1./n_input),
                                               size=(self.n_input+self.n_hidden, self.n_hidden)),
                             dtype=floatX)
        init_bi = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wi_r = theano.shared(value=init_Wi, name='Wi')
        self.bi_r = theano.shared(value=init_bi, name='bi')
        init_Wc = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                               high=np.sqrt(1./n_input),
                                               size=(self.n_input+self.n_hidden, self.n_hidden)),
                             dtype=floatX)
        init_bc = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wc_r = theano.shared(value=init_Wc, name='Wc')
        self.bc_r = theano.shared(value=init_bc, name='bc')
        init_Wo = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                               high=np.sqrt(1./n_input),
                                               size=(self.n_input+self.n_hidden, self.n_hidden)),
                             dtype=floatX)
        init_bo = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wo_r = theano.shared(value=init_Wo, name='Wo')
        self.bo_r = theano.shared(value=init_bo, name='bo')
        # parameters for column propagation
        init_Wf = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                               high=np.sqrt(1./n_input),
                                               size=(self.n_input+self.n_hidden*2, self.n_hidden)),
                             dtype=floatX)
        init_bf = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wf_c = theano.shared(value=init_Wf, name='Wf')
        self.bf_c = theano.shared(value=init_bf, name='bf')
        init_Wi = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                               high=np.sqrt(1./n_input),
                                               size=(self.n_input+self.n_hidden*2, self.n_hidden)),
                             dtype=floatX)
        init_bi = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wi_c = theano.shared(value=init_Wi, name='Wi')
        self.bi_c = theano.shared(value=init_bi, name='bi')
        init_Wc = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                               high=np.sqrt(1./n_input),
                                               size=(self.n_input+self.n_hidden*2, self.n_hidden)),
                             dtype=floatX)
        init_bc = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wc_c = theano.shared(value=init_Wc, name='Wc')
        self.bc_c = theano.shared(value=init_bc, name='bc')
        init_Wo = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                               high=np.sqrt(1./n_input),
                                               size=(self.n_input+self.n_hidden*2, self.n_hidden)),
                             dtype=floatX)
        init_bo = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wo_c = theano.shared(value=init_Wo, name='Wo')
        self.bo_c = theano.shared(value=init_bo, name='bo')
        # params
        self.params = [self.Wi_r, self.Wf_r, self.Wc_r, self.Wo_r, self.Wi_c, self.Wf_c, self.Wc_c, self.Wo_c,
                       self.bi_r, self.bf_r, self.bc_r, self.bo_r, self.bi_c, self.bf_c, self.bc_c, self.bo_c]
        self.build()

    def build(self):

        def _recurrence(x_t, m_r, m_c, h_tml_r, h_tml_c, c_tml_r, c_tml_c):
            x_er = self.Exr[x_t[:, 0], :]
            x_ec = self.Exc[x_t[:, 1], :]
            # row propagation independently
            concated_r = T.concatenate([x_er, h_tml_r], axis=-1)
            f_t_r = self.f(T.dot(concated_r, self.Wf_r) + self.bf_r)
            i_t_r = self.f(T.dot(concated_r, self.Wi_r) + self.bi_r)
            g_t_r = T.tanh(T.dot(concated_r, self.Wc_r) + self.bc_r)
            c_t_r = f_t_r * c_tml_r + i_t_r * g_t_r
            o_t_r = self.f(T.dot(concated_r, self.Wo_r) + self.bo_r)
            h_t_r = o_t_r * T.tanh(c_t_r)
            c_t_r = c_t_r * m_r[:, None]
            h_t_r = h_t_r * m_r[:, None]
            # column propagation with row information
            concated_c = T.concatenate([x_ec, h_tml_c, h_t_r], axis=-1)
            f_t_c = self.f(T.dot(concated_c, self.Wf_c) + self.bf_c)
            i_t_c = self.f(T.dot(concated_c, self.Wi_c) + self.bi_c)
            g_t_c = T.tanh(T.dot(concated_c, self.Wc_c) + self.bc_c)
            c_t_c = f_t_c * c_tml_c + i_t_c * g_t_c
            o_t_c = self.f(T.dot(concated_c, self.Wo_c) + self.bo_c)
            h_t_c = o_t_c * T.tanh(c_t_c)
            c_t_c = c_t_c * m_c[:, None]
            h_t_c = h_t_c * m_c[:, None]
            return h_t_r, h_t_c, c_t_r, c_t_c

        [h_r, h_c, c_r, c_c], update = theano.scan(fn=_recurrence,
                               sequences=[self.x, self.mask_r, self.mask_c],   # x.shape() = (n_maxlen, n_batch, 2)
                               outputs_info=[dict(initial=T.zeros((self.n_batch, self.n_hidden))),
                                             dict(initial=T.zeros((self.n_batch, self.n_hidden))),
                                             dict(initial=T.zeros((self.n_batch, self.n_hidden))),
                                             dict(initial=T.zeros((self.n_batch, self.n_hidden)))],
                               truncate_gradient=-1)

        h_r = T.concatenate([h_r[1:, :, :], T.zeros(shape=(1, self.n_batch, self.n_hidden))])
        # self.activation = [h_r, h_c]

        # dropout
        if self.p > 0:
            drop_mask_r = self.rng.binomial(n=1, p=1-self.p, size=h_r.shape, dtype=theano.config.floatX)
            drop_mask_c = self.rng.binomial(n=1, p=1-self.p, size=h_c.shape, dtype=theano.config.floatX)
            h_r = T.switch(T.eq(self.is_train, 1), h_r*drop_mask_r, h_r*(1-self.p))
            h_c = T.switch(T.eq(self.is_train, 1), h_c*drop_mask_c, h_c*(1-self.p))
        else:
            h_r = T.switch(T.eq(self.is_train, 1), h_r, h_r)
            h_c = T.switch(T.eq(self.is_train, 1), h_c, h_c)
        self.activation = [h_r, h_c]




