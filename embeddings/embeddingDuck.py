import duckdb
from os import path, makedirs, environ
import requests
import logging
from array import array
from io import StringIO

from time import time


class Embedding:

    @staticmethod
    def path(p):
        """

        Args:
            p (str): relative path.

        Returns:
            str: absolute path to the file, located in the ``$EMBEDDINGS_ROOT`` directory.

        """
        root = environ.get('EMBEDDINGS_ROOT', path.join(environ['HOME'], '.embeddings'))
        return path.join(path.abspath(root), p)

    @staticmethod
    def initialize_db(fname):
        """

        Args:
            fname (str): location of the database.

        Returns:
            db (sqlite3.Connection): a SQLite3 database with an embeddings table.

        """
        if path.dirname(fname) and not path.isdir(path.dirname(fname)):
            makedirs(path.dirname(fname))
        # open database in autocommit mode by setting isolation_level to None.

        db = duckdb.connect(fname)
        c = db.cursor()

        # Create query
        q_col = ' REAL, '.join(['d' + str( x) for x in range(300)]) + ' REAL'
        q = 'create table if not exists embeddings(word text primary key, {})'.format(q_col)
        c.execute(q)
        return db

    def __len__(self):
        """

        Returns:
            count (int): number of embeddings in the database.

        """
        c = self.db.cursor()
        q = c.execute('select count(*) from embeddings')
        return q.fetchone()[0]

    def insert_batch(self, batch):
        """

        Args:
            batch (list): a list of embeddings to insert, each of which is a tuple ``(word, embeddings)``.

        Example:

        .. code-block:: python

            e = Embedding()
            e.db = e.initialize_db(self.e.path('mydb.db'))
            e.insert_batch([
                ('hello', [1, 2, 3]),
                ('world', [2, 3, 4]),
                ('!', [3, 4, 5]),
            ])
        """
        c = self.db.cursor()
        binarized = [(word, emb[0]) + tuple(emb[1:]) for word, emb in batch]
        q = "insert into embeddings values (? {})".format(''.join([', ?' for x in range(300)]))
        try:
            c.execute("BEGIN TRANSACTION;")
            c.executemany(q, binarized)
            c.execute("COMMIT;")
        except Exception as e:
            print('insert failed\n{}'.format([w for w, e in batch]))
            raise e

    def __contains__(self, w):
        """

        Args:
            w: word to look up.

        Returns:
            whether an embedding for ``w`` exists.

        """
        return self.lookup(w) is not None

    def clear(self):
        """

        Deletes all embeddings from the database.

        """
        c = self.db.cursor()
        c.execute('delete from embeddings')

    def lookup(self, w):
        """

        Args:
            w: word to look up.

        Returns:
            embeddings for ``w``, if it exists.
            ``None``, otherwise.

        """
        c = self.db.cursor()
        q2 = ', '.join(['d' + str(x) for x in range(300)])
        q = c.execute("select {} from embeddings where word = '{}'".format(q2, w)).fetchone()
        return q

if __name__ == '__main__':
    from time import time

    file = '/mnt/c/Users/mickv/Desktop/enwiki-20190701-model-w2v-dim300'
    save_dir = '/mnt/c/Users/mickv/Desktop/'
    start = time()
    e = Embedding()
    db = e.initialize_db('/mnt/c/Users/mickv/Desktop/testdb.db')  # self.path(path.join('{}:{}.db'.format(name, d_emb))))