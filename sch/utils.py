import numpy as np
import cPickle as pkl
import math

def save_model(f, model):
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    pkl.dump(ps, open(f, 'wb'))


def load_model(f, model):
    ps = pkl.load(open(f, 'rb'))
    for p in model.params:
        p.set_value(ps[p.name])
    return model


class Vocab(object):
    def __init__(self, vocab_size):
        self.word_count = {'</s>': 0, '<UNK>': 0, '<s>': 0}
        self.words = None
        self.vocab_size = vocab_size

    def build_from_files(self, files):
        if type(files) is not list:
            raise ValueError("buildFromFiles input type error")

        print ("build vocabulary from files ...")
        for _file in files:
            line_num = 0
            for line in open(_file):
                line_num += 1
                for w in line.strip().split():
                    if w in self.word_count:
                        self.word_count[w] += 1
                    else:
                        self.word_count[w] = 1
            # self.word_count['<s>'] += line_num
            self.word_count['</s>'] += line_num
        count_pairs = sorted(self.word_count.items(), key=lambda x: (-x[1], x[0]))
        self.words, counts = list(zip(*count_pairs))

        #2 ids
        rows = int(math.ceil(math.sqrt(len(self.words))))
        cols = int(math.ceil(math.sqrt(len(self.words))))
        ids = []

        for row in xrange(rows):
            for col in xrange(cols):
                ids.append([row, col])

        final_ids = ids[0:len(self.words)]
        self.word2id = dict(zip(self.words, final_ids))
        self.UNK_ID = self.word2id['<UNK>']
        print ("vocab size: {}".format(self.size()))

    def encode(self, sentence):
        return [self.word2id[w] if self.word2id.has_key(w) else self.UNK_ID for w in sentence]

    def decode(self, ids):
        return [self.words[_id] for _id in ids]

    def size(self):
        return len(self.words)

    def word2id(self):
        return self.word2id


class TextIterator:
    def __init__(self, file, is_train, n_batch, maxlen=None, n_words_source=-1, vocabulary_size=None,
                 init_word_dict=None):
        self.file = file
        self.source = open(self.file, 'r')
        self.n_batch = n_batch
        self.maxlen = maxlen
        self.n_words_source = n_words_source
        self.end_of_data = False
        self.word_dict = None
        self.vocab_size = vocabulary_size

        if is_train and init_word_dict is None:
            self.vocab = Vocab(self.vocab_size)
            self.vocab.build_from_files([self.file])
            self.word_dict = self.vocab.word2id
        elif init_word_dict is not None:
            self.word_dict = init_word_dict

    def __iter__(self):
            return self

    def reset(self):
        self.source.seek(0)

    def set_word_dict(self, word_dict):
        self.word_dict = word_dict

    def get_word_dict(self):
        return self.word_dict

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        data = []
        bos_id = self.word_dict['<s>']
        unk_id = self.word_dict['<UNK>']
        eos_id = self.word_dict['</s>']
        try:
            while True:
                line = self.source.readline()
                if line == '':
                    raise IOError
                tokens = line.strip().split()
                tokens2id = [bos_id] + [bos_id] + \
                            [self.word_dict[w] if self.word_dict.has_key(w) else unk_id for w in tokens] + \
                            [eos_id]
                if self.maxlen and len(line) > self.maxlen:
                    continue
                data.append(tokens2id)
                if len(data) >= self.n_batch:
                    break
        except IOError:
            self.end_of_data = True

        if len(data) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        return prepare_data(data)


def prepare_data(seqs_x):
    lengths_x = [len(s) - 1 for s in seqs_x]
    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)

    x = np.zeros((maxlen_x, n_samples, 2)).astype('int32')
    y = np.zeros((maxlen_x, n_samples, 2)).astype('int32')
    y_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    x_mask_r = np.zeros((maxlen_x, n_samples)).astype('float32')
    x_mask_c = np.zeros((maxlen_x, n_samples)).astype('float32')

    for idx, s_x in enumerate(seqs_x):
        # row of x
        x[:lengths_x[idx], idx, 0] = [s_x[i][0] for i in range(0, lengths_x[idx])]
        # column of x
        x[:lengths_x[idx], idx, 1] = [s_x[i][1] for i in range(0, lengths_x[idx])]
        # row of y
        y[:lengths_x[idx], idx, 0] = [s_x[i][0] for i in range(1, lengths_x[idx]+1)]
        # column of y
        y[:lengths_x[idx], idx, 1] = [s_x[i][1] for i in range(1, lengths_x[idx]+1)]

        x_mask_r[:lengths_x[idx], idx] = 1
        x_mask_c[:lengths_x[idx], idx] = 1
        y_mask[:lengths_x[idx], idx] = 1

    return x, x_mask_r, x_mask_c, y, y_mask



