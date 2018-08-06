import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX


def sgd(params, gparams, lr=0.01):
    updates = []
    for p, gp in zip(params, gparams):
        updates.append((p, p-lr * gp))
    return updates


def nag(params, gparams, lr=0.001):
    raise NotImplementedError


def adagrad(params, gparams, lr=0.001):
    raise NotImplementedError


def rmsprop(params, gparams, lr=0.001, rho=0.9, epsilon=1e-6):
    '''
    Root Mean Square Propagation
    '''
    updates = []
    for p, g in zip(params, gparams):
        # g = T.clip(g, -grad_clip, grad_clip)
        r = theano.shared(p.get_value() * 0.)
        r_t = rho * r + (1 - rho) * g**2
        gradient_scaling = T.sqrt(r_t + epsilon)
        g = g / gradient_scaling
        updates.append((r, r_t))
        updates.append((p, p - lr * g))
    return updates


def adam(params, gparams, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    '''
    Adaptive Moment Estimation
    Reference: [ADAM: A Method for Stochastic Optimization.]
    '''
    updates = []
    i = theano.shared(np.dtype(theano.config.floatX).type(0))
    i_t = i + 1.
    fix1 = T.sqrt(1. - b2**i_t)
    fix2 = 1 - b1**i_t
    lr_t = lr * fix1 / fix2
    for p, g in zip(params, gparams):

        # g = T.clip(g, -grad_clip, grad_clip)
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = b1 * m + (1. - b1) * g
        v_t = b2 * v + (1. - b2) * T.sqr(g)
        M_t = m_t / (1 - b1**i_t)
        V_t = v_t / (1 - b2**i_t)
        p_t = p - lr_t * M_t / (T.sqrt(V_t) + e)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


