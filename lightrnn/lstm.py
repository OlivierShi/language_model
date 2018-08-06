import numpy as np
import theano
from theano import tensor as T

class RNN:
    def __init__(self,rng,
                 n_input,n_hidden,n_batch,
                 x,Exr,Exc,mask_r, mask_c, is_train=1, p=0.5):
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
        # forget gate
        init_Wf = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                              high=np.sqrt(1./n_input),
                                              size=(self.n_input+self.n_hidden, self.n_hidden)),
                            dtype=floatX)
        init_bf = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wf = theano.shared(value=init_Wf, name='Wf')
        self.bf = theano.shared(value=init_bf, name='bf')
        # input gate
        init_Wi = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                              high=np.sqrt(1./n_input),
                                              size=(self.n_input+self.n_hidden, self.n_hidden)),
                            dtype=floatX)
        init_bi = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wi = theano.shared(value=init_Wi, name='Wi')
        self.bi = theano.shared(value=init_bi, name='bi')
        # cell gate
        init_Wc = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                              high=np.sqrt(1./n_input),
                                              size=(self.n_input+self.n_hidden, self.n_hidden)),
                            dtype=floatX)
        init_bc = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wc = theano.shared(value=init_Wc, name='Wc')
        self.bc = theano.shared(value=init_bc, name='bc')
        # output gate
        init_Wo = np.asarray(np.random.uniform(low=-np.sqrt(1./n_input),
                                              high=np.sqrt(1./n_input),
                                              size=(self.n_input+self.n_hidden, self.n_hidden)),
                            dtype=floatX)
        init_bo = np.asarray(np.zeros(self.n_hidden), dtype=floatX)
        self.Wo = theano.shared(value=init_Wo, name='Wo')
        self.bo = theano.shared(value=init_bo, name='bo')
        # params
        self.params = [self.Wi, self.Wf, self.Wc, self.Wo,
                       self.bi, self.bf, self.bc, self.bo]
        self.build()

    def build(self):

        def _recurrence(x_t, m_r, m_c, h_tml_r, h_tml_c, c_tml_r, c_tml_c):
            x_er = self.Exr[x_t[:, 0], :]
            x_ec = self.Exc[x_t[:, 1], :]
            # row
            concated_r = T.concatenate([x_er, h_tml_c], axis=-1)
            # forget gate
            f_t_r = self.f(T.dot(concated_r, self.Wf) + self.bf)
            # input gate
            i_t_r = self.f(T.dot(concated_r, self.Wi) + self.bi)
            # cell update
            g_t_r = T.tanh(T.dot(concated_r, self.Wc) + self.bc)
            c_t_r = f_t_r * c_tml_c + i_t_r * g_t_r
            # output gate
            o_t_r = self.f(T.dot(concated_r, self.Wo) + self.bo)
            # hidden state
            h_t_r = o_t_r * T.tanh(c_t_r)
            c_t_r = c_t_r * m_r[:, None]
            h_t_r = h_t_r * m_r[:, None]
            # column
            concated_c = T.concatenate([x_ec, h_t_r], axis=-1)
            # forget gate
            f_t_c = self.f(T.dot(concated_c, self.Wf) + self.bf)
            # input gate
            i_t_c = self.f(T.dot(concated_c, self.Wi) + self.bi)
            # cell update
            g_t_c = T.tanh(T.dot(concated_c, self.Wc) + self.bc)
            c_t_c = f_t_c * c_t_r + i_t_c * g_t_c
            # output gate
            o_t_c = self.f(T.dot(concated_c, self.Wo) + self.bo)
            # hidden state
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




