from os.path import join, dirname

from torchtext.data import Field, TabularDataset, Iterator

_DATA_DIR = join(dirname(dirname(__file__)), 'data')


def load_dataset(file_name):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset.
    """
    file_path = join(_DATA_DIR, file_name)
    label_field = Field(sequential=False, use_vocab=False)
    text_field = Field()

    return TabularDataset(
        path=file_path,
        format='csv',
        fields=[('doc_id', label_field), ('text', text_field)])


class NCEIterator(Iterator):
    """An iterator for noise-contrastive estimation of word vector models.

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

    Remaining keyword arguments:
        Passed to the constructor of the iterator class being used.
    """
    def __init__(self, dataset, batch_size, context_size,
                 num_noise_words, **kwargs):
        self.context_size = context_size
        self.num_noise_words = num_noise_words
        super(NCEIterator, self).__init__(dataset, batch_size, **kwargs)

    def __iter__(self):
        # todo
        pass
