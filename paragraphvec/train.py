import sys
from os.path import join, dirname
from shutil import copyfile

import fire
import torch
from torch.optim import SGD

from paragraphvec.data import load_dataset, NCEGenerator
from paragraphvec.loss import NegativeSampling
from paragraphvec.models import DistributedMemory

_MODELS_DIR = join(dirname(dirname(__file__)), 'models')


def start(data_file_name,
          num_epochs,
          batch_size,
          context_size,
          num_noise_words,
          vec_dim,
          lr):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory.

    num_epochs: int
        Number of iterations to train the model (i.e. number
        of times every example is seen during training).

    batch_size: int
        Number of examples per single gradient update.

    context_size: int
        Half the size of a neighbourhood of target words (i.e. how many
        words left and right are regarded as context).

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    lr: float
        Learning rate of the SGD optimizer (uses 0.9 nesterov momentum).
    """
    dataset = load_dataset(data_file_name)
    data_generator = NCEGenerator(
        dataset,
        batch_size,
        context_size,
        num_noise_words)

    model = DistributedMemory(
        vec_dim,
        num_docs=len(dataset),
        num_words=data_generator.vocabulary_size())

    cost_func = NegativeSampling()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    if torch.cuda.is_available():
        model.cuda()

    print("Dataset comprised of {:d} documents.".format(len(dataset)))
    print("Vocabulary size is {:d}.\n".format(data_generator.vocabulary_size()))

    best_loss = sys.float_info.max

    for epoch in range(num_epochs):
        print("Epoch {:d}".format(epoch + 1), end='')

        loss = []
        for _ in range(len(data_generator)):
            batch = data_generator.next()

            x = model.forward(
                batch.context_ids,
                batch.doc_ids,
                batch.target_noise_ids)
            x = cost_func.forward(x)

            loss.append(x.data[0])

            model.zero_grad()
            x.backward()
            optimizer.step()

        loss = torch.mean(torch.FloatTensor(loss))
        print(" - loss: {:.4f}".format(loss))

        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        _save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()}, is_best_loss)


def _save_checkpoint(state, is_best_loss, file_name='checkpoint.pth.tar'):
    file_path = join(_MODELS_DIR, file_name)
    torch.save(state, file_path)
    if is_best_loss:
        copyfile(file_path, join(_MODELS_DIR, 'model_best.pth.tar'))


if __name__ == '__main__':
    fire.Fire()
