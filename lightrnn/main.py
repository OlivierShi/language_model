import time
from rnnlm import *
from utils import TextIterator, save_model, calculate_wer
from argparse import ArgumentParser
import sys
from updates import *
import mcmf
import collections

lr = 0.001
p = 0.5
NEPOCH = 200

n_input = 256
n_hidden = 1024
optimizer = adam

argument = ArgumentParser(usage='it is usage tip', description='no')
argument.add_argument('--train_file', default='../data/wikitext-103/wiki.train.txt', type=str, help='train dir')
argument.add_argument('--valid_file', default='../data/wikitext-103/wiki.valid.txt', type=str, help='valid dir')
argument.add_argument('--test_file', default='../data/wikitext-103/wiki.test.txt', type=str, help='test dir')
argument.add_argument('--vocab_size', default=267736, type=int, help='vocab size')
argument.add_argument('--batch_size', default=40, type=int, help='batch size')

args = argument.parse_args()

train_datafile = args.train_file
valid_datafile = args.valid_file
test_datafile = args.test_file
n_batch = args.batch_size
vocabulary_size = args.vocab_size
# train_datafile = 'data/idx_2_ptb.train.txt'
# valid_datafile = 'data/idx_2_ptb.valid.txt'
# test_datafile = 'data/idx_2_ptb.test.txt'
# n_batch = 10
# vocabulary_size = 10001
n_words_source = -1
# ptb
#disp_freq = 40
#valid_freq = 1000
#test_freq = 2000
#save_freq = 20000
#clip_freq = 2000
#pred_freq = 20000
word_allocation_freq = 20

# wiki-103
disp_freq = 1000
valid_freq = 10000
test_freq = 20000
save_freq = 800000
clip_freq = 40000
pred_freq = 4000000

E_table_size = int(np.ceil(np.sqrt(vocabulary_size)))

def evaluate(test_data, model):
    cost = 0.
    index = 0.
    sumed_wer_r = []
    sumed_wer_c = []

    for x, x_mask_r, x_mask_c, y, y_mask in test_data:
        y_r = y[:, :, 0]
        y_c = y[:, :, 1]
        index += np.sum(y_mask)
        test_model =  model.test(x, x_mask_r, x_mask_c, y, y_mask, x.shape[1])
        cost += test_model[0]
        pred_r, pred_c = test_model[1], test_model[2]
        # wer
        sumed_wer_r.append(calculate_wer(y_r, y_mask, np.reshape(pred_r, y_r.shape)))
        sumed_wer_c.append(calculate_wer(y_c, y_mask, np.reshape(pred_c, y_c.shape)))

    return cost / index, np.sum(sumed_wer_c + sumed_wer_r) / (index * 2)


def train(lr):
    # load data
    print 'loading dataset...'
    train_data = TextIterator(train_datafile, is_train=True, n_words_source=n_words_source, n_batch=n_batch,
                              vocabulary_size=vocabulary_size)
    word_dict = train_data.get_word_dict()
    valid_data = TextIterator(valid_datafile, is_train=False, n_words_source=n_words_source, n_batch=n_batch)
    valid_data.set_word_dict(word_dict)
    test_data = TextIterator(test_datafile, is_train=False, n_words_source=n_words_source, n_batch=n_batch)
    test_data.set_word_dict(word_dict)

    print 'building model...'
    model = RNNLM(n_input, n_hidden, vocabulary_size, optimizer)

    print 'training start...'
    start = time.time()
    idx = 0
    for epoch in xrange(1, NEPOCH + 1):
        error = 0.
        err_norm = 0.
        epoch_time = time.time()
        for x, x_mask_r, x_mask_c, y, y_mask in train_data:
            idx += 1
            err_norm += np.sum(y_mask)
            cost = model.train(x, x_mask_r, x_mask_c, y, y_mask, x.shape[1], lr)
            error += cost
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
                valid_cost, wer = evaluate(valid_data, model)
                print 'valid cost: ', valid_cost, 'perplexity: ', np.exp(valid_cost), 'wer: ', wer
            if idx % test_freq == 0:
                print 'testing...'
                test_cost, wer = evaluate(test_data, model)
                print 'test cost: ', test_cost, 'perplexity: ', np.exp(test_cost), 'wer: ', wer
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
	    alloc_time = time.time()
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
                logit_shape = x.shape
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
            print "time consuming for word re-allocation preparing: " + str(time.time() - alloc_time)
            print "re-allocation starting..."
            word_dict = mcmf.MCMF(word_dict, cost_dict_r, cost_dict_c)
            print "re-allocation total time: " + str(time.time() - alloc_time)
            print "resetting the word dictionary..."
            train_data.set_word_dict(word_dict)
            valid_data.set_word_dict(word_dict)
            test_data.set_word_dict(word_dict)

    print "Finished. Time = " + str(time.time() - start)


if __name__ == '__main__':
    train(lr=lr)

