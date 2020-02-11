from embeddings.embedding import Embedding
# from embeddings.embeddingDuck import Embedding

from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, frombuffer
import numpy as np
from gensim import utils

class GenericEmbedding(Embedding):
    """
    Reference: http://nlp.stanford.edu/projects/glove
    """
    def __init__(self, name, save_dir, file_name, d_emb=300, categories=[], default='none', reset=False, batch_size=5000):
        """
        Args:
            name: name of the embedding to retrieve.
            d_emb: embedding dimensions.
            show_progress: whether to print progress.
            default: how to embed words that are out of vocabulary. Can use zeros, return ``None``, or generate random between ``[-0.1, 0.1]``.
        """
        self.file_name = file_name
        self.avg_cnt = {'word': {'cnt': 0, 'sum': zeros(d_emb)}}
        for c in categories:
            self.avg_cnt[c] =  {'cnt': 0, 'sum': zeros(d_emb)}

        # self.setting = self.settings[name]
        assert default in {'none', 'random', 'zero'}

        path_db = '{}/{}.db'.format(save_dir, name)

        self.categories = categories
        self.d_emb = d_emb
        self.name = name
        self.db = self.initialize_db(path_db)#self.path(path.join('{}:{}.db'.format(name, d_emb))))
        self.default = default
        self.batch_size = batch_size

        if reset:
            self.seen = set()
            self.clear()
            self.load_word2emb()

    def emb(self, word, default=None):
        g = self.lookup(word)
        return g

    def load_word2emb(self):
        unicode_errors = 'strict'
        encoding = 'utf-8'
        # fin_name = self.ensure_file(path.join('glove', '{}.zip'.format(self.name)), url=self.setting.url)
        batch = []
        start = time()

        limit = 5000

        # Loop over file.
        with utils.open(self.file_name, 'rb') as fin:
            # Determine size file.
            header = utils.to_unicode(fin.readline(), encoding='utf-8')
            vocab_size, vector_size = (int(x) for x in header.split())  # throws for invalid file format

            if limit:
                vocab_size = min(vocab_size, limit)

            for line_no in range(vocab_size):
                line = fin.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")

                parts = utils.to_unicode(line.rstrip(), encoding=encoding, errors=unicode_errors).split(" ")

                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % line_no)

                word, vec = parts[0], np.array([float(x) for x in parts[1:]])

                if word in self.seen:
                    continue

                self.seen.add(word)
                batch.append((word, vec))


                #TODO:
                # cat_subset = [c for c in self.categories if c in word]
                # if len(cat_subset) == 1:
                #     cat = cat_subset[0]
                #
                #     self.avg_cnt[cat]['cnt'] += 1
                #     self.avg_cnt[cat]['sum'] += vec
                # else:
                #     self.avg_cnt['word']['cnt'] += 1
                #     self.avg_cnt['word']['sum'] += vec

                # if len(batch) == self.batch_size:
                #     print('Another {}'.format(self.batch_size), line_no, time() - start)
                #     start = time()
                #     print("trying to insert batch")
                #     self.insert_batch(batch)
                #     batch.clear()

        for c in self.categories:
            if self.avg_cnt[c]['cnt'] > 0:
                batch.append(('#{}UNK#'.format(c), self.avg_cnt[c]['sum'] / self.avg_cnt[c]['cnt']))
                print('Added average for category: #{}UNK#'.format(c))

        if self.avg_cnt['word']['cnt'] > 0:
            batch.append(('#WORD/UNK#', self.avg_cnt['word']['sum'] / self.avg_cnt['word']['cnt']))
            print('Added average for category: #WORD/UNK#'.format(c))

        if batch:
            self.insert_batch(batch)

if __name__ == '__main__':
    from time import time
    file = '/mnt/c/Users/mickv/Desktop/enwiki-20190701-model-w2v-dim300'
    save_dir = '/mnt/c/Users/mickv/Desktop/'
    start = time()

    emb = GenericEmbedding('wiki2vec2019', save_dir, file, d_emb=300, categories=['ENTITY/'], reset=True)
    print('loading 5k', time() - start)
    try_set = ['in'] * 1000
    times = []
    for t in try_set:
        start = time()
        emb.emb(t)
        times.append(time() - start)

    print(np.mean(times))