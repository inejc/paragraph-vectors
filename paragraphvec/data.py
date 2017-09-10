import threading
from math import ceil
from os.path import join, dirname

from torchtext.data import Field, TabularDataset

_DATA_DIR = join(dirname(dirname(__file__)), 'data')


def load_dataset(file_name):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset.
    """
    file_path = join(_DATA_DIR, file_name)
    text_field = Field(pad_token=None)

    dataset = TabularDataset(
        path=file_path,
        format='csv',
        fields=[('text', text_field)])

    text_field.build_vocab(dataset)
    return dataset


class NCEGenerator(object):
    """A thread-safe batch generator for noise-contrastive estimation
    of word vector models.

    Parameters
    ----------
    dataset: torchtext.data.Dataset
        todo.

    batch_size: int
        todo.

    context_size: int
        todo.

    num_noise_words: int
        todo.
    """
    def __init__(self, dataset, batch_size, context_size, num_noise_words):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_noise_words = num_noise_words
        self.lock = threading.Lock()
        # document id and in-document position define
        # the current indexing state
        self.doc_id = 0
        self.in_doc_pos = self.context_size

    def __len__(self):
        num_examples = sum(self._num_examples_in_doc(d) for d in self.dataset)
        return ceil(num_examples / self.batch_size)

    def __iter__(self):
        while True:
            # advance indices in a thread-safe manner
            with self.lock:
                doc_id = self.doc_id
                in_doc_pos = self.in_doc_pos
                self._advance_indices()

            # todo
            pass

    def _advance_indices(self):
        num_examples = self._num_examples_in_doc(
            self.dataset[self.doc_id], self.in_doc_pos)

        if num_examples >= self.batch_size:
            # enough examples in the current document
            self.in_doc_pos += self.batch_size
            return

        while num_examples < self.batch_size:
            if self.doc_id == len(self.dataset) - 1:
                # last document: reset indices
                self.doc_id = 0
                self.in_doc_pos = self.context_size
                return

            self.doc_id += 1
            num_examples += self._num_examples_in_doc(self.dataset[self.doc_id])

        self.in_doc_pos = (len(self.dataset[self.doc_id].text) - 1
                           - self.context_size
                           - (num_examples - self.batch_size))

    def _num_examples_in_doc(self, doc, in_doc_pos=None):
        if in_doc_pos is not None:
            # number of remaining
            if len(doc.text) - in_doc_pos >= self.context_size + 1:
                return len(doc.text) - in_doc_pos - self.context_size
            return 0

        if len(doc.text) >= 2 * self.context_size + 1:
            # total number
            return len(doc.text) - 2 * self.context_size
        return 0
