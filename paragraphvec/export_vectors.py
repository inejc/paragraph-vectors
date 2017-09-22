import csv
import re
from os.path import join

import fire
import torch

from paragraphvec.data import load_dataset
from paragraphvec.models import DistributedMemory
from paragraphvec.utils import DATA_DIR, MODELS_DIR


def start(data_file_name, model_file_name):
    """Saves trained paragraph vectors to a csv file in the *data* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory that was used during training.

    model_file_name: str
        Name of a file in the *models* directory (a model trained on
        the *data_file_name* dataset).
    """
    dataset = load_dataset(data_file_name)
    model = _load_model(
        model_file_name,
        num_docs=len(dataset),
        num_words=len(dataset.fields['text'].vocab) - 1)

    def qm(str_): return '\"' + str_ + '\"'

    result_lines = []

    with open(join(DATA_DIR, data_file_name)) as file:
        lines = csv.reader(file)
        for i, line in enumerate(lines):
            result_line = [qm(x) if not x.isnumeric() else x for x in line[1:]]
            result_line += [str(x) for x in model.get_paragraph_vector(i)]
            result_lines.append(','.join(result_line) + '\n')

    result_file_name = model_file_name[:-7] + 'csv'

    with open(join(DATA_DIR, result_file_name), 'w') as f:
        f.writelines(result_lines)


def _load_model(model_file_name, num_docs, num_words):
    vec_dim = int(re.search('_vecdim\.(\d+)_', model_file_name).group(1))
    model_file_path = join(MODELS_DIR, model_file_name)

    try:
        checkpoint = torch.load(model_file_path)
    except AssertionError:
        checkpoint = torch.load(
            model_file_path,
            map_location=lambda storage, location: storage)

    model = DistributedMemory(vec_dim, num_docs, num_words)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == '__main__':
    fire.Fire()
