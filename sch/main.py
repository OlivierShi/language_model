import time
from rnnlm import *
from utils import TextIterator, save_model
from argparse import ArgumentParser
import sys
from updates import *
import mcmf
import cPickle as pkl
import collections

lr = 0.001
p = 0.5
NEPOCH = 200

n_input = 256
n_hidden = 1000
optimizer = adam
#optimizer = rmsprop

argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--train_file', default='../data/oneb/ob.train.txt', type=str, help='train dir')
argument.add_argument('--valid_file', default='../data/oneb/ob.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/oneb/ob.test.txt', type=str, help='test dir')
argument.add_argument('--vocab_size', default=793471, type=int, help='vocab size')
argument.add_argument('--batch_size', default=20, type=int, help='batch size')
argument.add_argument('--disp_freq', default=4000, type=int, help='disp freq')
argument.add_argument('--valid_freq', default=100000, type=int, help='valid freq')
argument.add_argument('--test_freq', default=200000, type=int, help='test freq')
argument.add_argument('--save_freq', default=1000000, type=int, help='save freq')
argument.add_argument('--clip_freq', default=750000, type=int, help='clip freq')
argument.add_argument('--word_allocation_freq', default=20, type=int, help='word allocation freq')
argument.add_argument('--init_word_dict_file', default='../data/oneb/word_dicts/init_word_dict_890.pkl', type=str, help='initial word dict')
cluster_num = 890
#argument.add_argument('--train_file', default='data/ptb.train.txt', type=str, help='train dir')
#argument.add_argument('--valid_file', default='data/ptb.valid.txt', type=str, help='valid dir')
#argument.add_argument('--test_file', default='data/ptb.test.txt', type=str, help='test dir')
#argument.add_argument('--vocab_size', default=10001, type=int, help='vocab size')
#argument.add_argument('--batch_size', default=10, type=int, help='batch size')
#argument.add_argument('--disp_freq', default=40, type=int, help='disp freq')
#argument.add_argument('--valid_freq', default=1000, type=int, help='valid freq')
#argument.add_argument('--test_freq', default=2000, type=int, help='test freq')
#argument.add_argument('--save_freq', default=20000, type=int, help='save freq')
#argument.add_argument('--clip_freq', default=2000, type=int, help='clip freq')
#argument.add_argument('--word_allocation_freq', default=20, type=int, help='word allocation freq')
#argument.add_argument('--init_word_dict_file', default='data/word_dicts/init_word_dict_100.pkl', type=str, help='initial word dict')
#argument.add_argument('--train_file', default='../data/wikitext-103/wiki.train.txt', type=str, help='train dir')
#argument.add_argument('--valid_file', default='../data/wikitext-103/wiki.valid.txt', type=str, help='valid dir')
#argument.add_argument('--test_file', default='../data/wikitext-103/wiki.test.txt', type=str, help='test dir')
#argument.add_argument('--vocab_size', default=267736, type=int, help='vocab size')
#argument.add_argument('--batch_size', default=40, type=int, help='batch size')
#argument.add_argument('--disp_freq', default=1000, type=int, help='disp freq')
#argument.add_argument('--valid_freq', default=10000, type=int, help='valid freq')
#argument.add_argument('--test_freq', default=20000, type=int, help='test freq')
#argument.add_argument('--save_freq', default=800000, type=int, help='save freq')
#argument.add_argument('--clip_freq', default=40000, type=int, help='clip freq')
#argument.add_argument('--word_allocation_freq', default=10, type=int, help='word allocation freq')
#argument.add_argument('--init_word_dict_file', default='../data/wikitext-103/init_word_dict.pkl', type=str, help='initial word dict')


args = argument.parse_args()

train_datafile = args.train_file
valid_datafile = args.valid_file
test_datafile = args.test_file
n_batch = args.batch_size
vocabulary_size = args.vocab_size
disp_freq = args.disp_freq
valid_freq = args.valid_freq
test_freq = args.test_freq
save_freq = args.save_freq
clip_freq = args.clip_freq
word_allocation_freq = args.word_allocation_freq
init_word_dict_file = args.init_word_dict_file

with open(init_word_dict_file, 'rb') as f:
    init_word_dict = pkl.load(f)

n_words_source = -1
# ptb
#disp_freq = 40
#valid_freq = 1000
#test_freq = 2000
#save_freq = 20000
#clip_freq = 2000
#pred_freq = 20000
# wiki-103
#disp_freq = 1000
#valid_freq = 10000
#test_freq = 20000
#save_freq = 800000
#clip_freq = 40000
#word_allocation_freq = 20
E_table_size = int(np.ceil(np.sqrt(vocabulary_size)))


def evaluate(test_data, model):
    cost = 0.
    index = 0
    for x, x_mask_r, x_mask_c, y, y_mask in test_data:
        index += np.sum(y_mask)
        cost += model.test(x, x_mask_r, x_mask_c, y, y_mask, x.shape[1])
    return cost / index


def train(lr):
    # load data
    print 'loading dataset...'
    train_data = TextIterator(train_datafile, is_train=True, n_words_source=n_words_source, n_batch=n_batch,
                              vocabulary_size=vocabulary_size, init_word_dict=init_word_dict)
    word_dict = train_data.get_word_dict()
    valid_data = TextIterator(valid_datafile, is_train=False, n_words_source=n_words_source, n_batch=n_batch)
    valid_data.set_word_dict(word_dict)
    test_data = TextIterator(test_datafile, is_train=False, n_words_source=n_words_source, n_batch=n_batch)
    test_data.set_word_dict(word_dict)

    print 'building model...'
    model = RNNLM(n_input, n_hidden, vocabulary_size, cluster_num, optimizer, p)

    print 'training start...'
    start = time.time()
    idx = 0
    err_norm = 0.
    for epoch in xrange(1, NEPOCH):
        error = 0.
        epoch_time = time.time()
        for x, x_mask_r, x_mask_c, y, y_mask in train_data:
            idx += 1
            err_norm += np.sum(y_mask)
            cost = model.train(x, x_mask_r, x_mask_c, y, y_mask, x.shape[1], lr)[0]
            error += np.sum(cost)
            if np.isnan(cost) or np.isinf(cost):
                print 'NaN or Inf detected!'
                return -1
            if idx % disp_freq == 0:
                print 'epoch: ', epoch, 'idx: ', idx, 'cost: ', error/err_norm, 'ppl: ', np.exp(error/err_norm), 'lr: ', lr
                error = 0.
                err_norm = 0.
            if idx % save_freq == 0:
                print 'dumping...'
                save_model('model/parameters_%.2f.pkl'%(time.time()-start), model)
            if idx % valid_freq == 0:
                print 'validing...'
                valid_cost = evaluate(valid_data, model)
                print 'valid cost: ', valid_cost, 'perplexity: ', np.exp(valid_cost)
            if idx % test_freq == 0:
                print 'testing...'
                test_cost = evaluate(test_data, model)
                print 'test cost: ', test_cost, 'perplexity: ', np.exp(test_cost)
            # if idx % pred_freq == 0:
            #     print 'predicting...'
            #     prediction = model.predict(x, x_mask_r, x_mask_c, x.shape[1])
            #     print prediction[:100]
            # if idx % clip_freq == 0 and lr >= 1e-3:
            #     lr = lr * 0.7
            #     print 'cliping learning rate: %f' % lr
        print 'epoch %d, time consuming: %f' % (epoch, time.time() - epoch_time)
        sys.stdout.flush()
        if epoch % word_allocation_freq == 0:
            # word re-allocation
            print 'word re-allocation preparing...'
            cost_dict_r = collections.defaultdict()
            cost_dict_c = collections.defaultdict()
            # cost_table = np.zeros((E_table_size**2, E_table_size**2))
            word_freq = np.zeros(E_table_size**2)
            _idx = 0
            for x, x_mask_r, x_mask_c, y, y_mask in train_data:
                _idx += 1

                softmax_r, softmax_c = model.getNLL(x, x_mask_r, x_mask_c, x.shape[1])

                word = np.reshape(y, (-1, 2))
                for index, position in enumerate(word):
                    if tuple(position) in cost_dict_r:
                        cost_dict_r[tuple(position)] += softmax_r[index]
                        cost_dict_c[tuple(position)] += softmax_c[index]
                    else:
                        cost_dict_r[tuple(position)] = softmax_r[index]
                        cost_dict_c[tuple(position)] = softmax_c[index]

                # x_r = x[:, :, 0].reshape([logit_shape[0] * logit_shape[1]])
                # x_c = x[:, :, 1].reshape([logit_shape[0] * logit_shape[1]])
                # for w in range(logit_shape[0] * logit_shape[1]):
                #     idx_in_vocab = x_r[w] * E_table_size + x_c[w]
                #     word_freq[idx_in_vocab] += 1
                #     cost_table[idx_in_vocab, :] += softmax_r[w].repeat(E_table_size) + np.tile(softmax_c[w],
                #                                                                                reps=E_table_size)
                #     # cost_table_r[idx_in_vocab, :] += softmax_r[i, :]
                #     # cost_table_c[idx_in_vocab, :] += softmax_c[i, :]
                # print "batch number %d of train data is added into re-allocation table" % _idx
            # start re-allocation
            print "re-allocation starting..."
            word_dict = mcmf.MCMF(word_dict, cost_dict_r, cost_dict_c)
            print "resetting the word dictionary..."
            train_data.set_word_dict(word_dict)
            valid_data.set_word_dict(word_dict)
            test_data.set_word_dict(word_dict)

    print "Finished. Time = " + str(time.time() - start)


if __name__ == '__main__':
    train(lr=lr)

