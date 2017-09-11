import threading
from math import ceil
from os.path import join, dirname

import torch
from torchtext.data import Field, TabularDataset

_DATA_DIR = join(dirname(dirname(__file__)), 'data')


def load_dataset(file_name):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset instance.
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
    """An infinite, thread-safe batch generator for noise-contrastive
    estimation of word vector models.

    Parameters
    ----------
    dataset: torchtext.data.TabularDataset
        Dataset from which examples are generated. A column labeled *text*
        is expected and should be comprised of a list of tokens. Each row
        should represent a single document.

    batch_size: int
        Number of examples per single gradient update.

    context_size: int
        Half the size of a neighbourhood of target words (i.e. how many
        words left and right are regarded as context).

    num_noise_words: int
        Number of noise words to sample from the noise distribution.
    """
    def __init__(self, dataset, batch_size, context_size, num_noise_words):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_noise_words = num_noise_words

        self._vocabulary = self.dataset.fields['text'].vocab
        self._noise = torch.Tensor(len(self._vocabulary) - 1).zero_()
        self._init_noise_distribution()

        # document id and in-document position define
        # the current indexing state
        self._lock = threading.Lock()
        self._doc_id = 0
        self._in_doc_pos = self.context_size

    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        for word, freq in self._vocabulary.freqs.items():
            self._noise[self._word_to_index(word)] = freq
        self._noise.pow_(0.75)

    def __len__(self):
        num_examples = sum(self._num_examples_in_doc(d) for d in self.dataset)
        return ceil(num_examples / self.batch_size)

    def vocabulary_size(self):
        return len(self._vocabulary) - 1

    def next(self):
        """Generates the next batch of examples in a thread-safe manner."""
        with self._lock:
            doc_id = self._doc_id
            in_doc_pos = self._in_doc_pos
            self._advance_indices()

        # generate the actual batch
        batch = NCEBatch()

        while len(batch) < self.batch_size:
            if doc_id == len(self.dataset):
                # last document exhausted
                return self._batch_to_torch_data(batch)
            if in_doc_pos <= (len(self.dataset[doc_id].text) - 1
                              - self.context_size):
                # more examples in the current document
                self._add_example_to_batch(doc_id, in_doc_pos, batch)
                in_doc_pos += 1
            else:
                # go to the next document
                doc_id += 1
                in_doc_pos = self.context_size

        return self._batch_to_torch_data(batch)

    def _advance_indices(self):
        num_examples = self._num_examples_in_doc(
            self.dataset[self._doc_id], self._in_doc_pos)

        if num_examples > self.batch_size:
            # more examples in the current document
            self._in_doc_pos += self.batch_size
            return

        if num_examples == self.batch_size:
            # just enough examples in the current document
            if self._doc_id < len(self.dataset) - 1:
                self._doc_id += 1
            else:
                self._doc_id = 0
            self._in_doc_pos = self.context_size
            return

        while num_examples < self.batch_size:
            if self._doc_id == len(self.dataset) - 1:
                # last document: reset indices
                self._doc_id = 0
                self._in_doc_pos = self.context_size
                return

            self._doc_id += 1
            num_examples += self._num_examples_in_doc(
                self.dataset[self._doc_id])

        self._in_doc_pos = (len(self.dataset[self._doc_id].text)
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

    def _add_example_to_batch(self, doc_id, in_doc_pos, batch):
        doc = self.dataset[doc_id].text
        batch.doc_ids.append(doc_id)

        current_context = []
        for i in range(-self.context_size, self.context_size + 1):
            if i != 0:
                current_context.append(self._word_to_index(doc[in_doc_pos - i]))
        batch.context_ids.append(current_context)

        # sample from the noise distribution
        current_noise = torch.multinomial(
            self._noise,
            self.num_noise_words,
            replacement=True).tolist()
        current_noise.insert(0, self._word_to_index(doc[in_doc_pos]))
        batch.target_noise_ids.append(current_noise)

    def _word_to_index(self, word):
        return self._vocabulary.stoi[word] - 1

    @staticmethod
    def _batch_to_torch_data(batch):
        batch.context_ids = torch.LongTensor(batch.context_ids)
        batch.doc_ids = torch.LongTensor(batch.doc_ids)
        batch.target_noise_ids = torch.LongTensor(batch.target_noise_ids)

        if torch.cuda.is_available():
            batch.context_ids.cuda()
            batch.doc_ids.cuda()
            batch.target_noise_ids.cuda()

        return batch


class NCEBatch(object):
    def __init__(self):
        self.context_ids = []
        self.doc_ids = []
        self.target_noise_ids = []

    def __len__(self):
        return len(self.doc_ids)
