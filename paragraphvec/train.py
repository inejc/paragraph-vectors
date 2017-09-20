import time
from os import remove
from os.path import join
from sys import stdout, float_info

import fire
import torch
from torch.optim import SGD

from paragraphvec.data import load_dataset, NCEData
from paragraphvec.loss import NegativeSampling
from paragraphvec.models import DistributedMemory
from paragraphvec.utils import MODELS_DIR, MODEL_NAME


def start(data_file_name,
          context_size,
          num_noise_words,
          vec_dim,
          num_epochs,
          batch_size,
          lr,
          model_ver='dm',
          vec_combine_method='sum',
          save_all=False,
          max_generated_batches=5,
          num_workers=1):
    """Trains a new model. The latest checkpoint and the best performing
    model are saved in the *models* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory.

    context_size: int
        Half the size of a neighbourhood of target words (i.e. how many
        words left and right are regarded as context).

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    vec_dim: int
        Dimensionality of vectors to be learned (for paragraphs and words).

    num_epochs: int
        Number of iterations to train the model (i.e. number
        of times every example is seen during training).

    batch_size: int
        Number of examples per single gradient update.

    lr: float
        Learning rate of the SGD optimizer (uses 0.9 nesterov momentum).

    model_ver: str, one of ('dm', 'dbow'), default='dm'
        Version of the model as proposed by Q. V. Le et al., Distributed
        Representations of Sentences and Documents. 'dm' stands for
        Distributed Memory, 'dbow' stands for Distributed Bag Of Words.
        Currently only the 'dm' version is implemented.

    vec_combine_method: str, one of ('sum', 'concat'), default='sum'
        Method for combining paragraph and word vectors in the 'dm' model.
        Currently only the 'sum' operation is implemented.

    save_all: bool, default=False
        Indicates whether a checkpoint is saved after each epoch.
        If false, only the best performing model is saved.

    max_generated_batches: int, default=5
        Maximum number of pre-generated batches.

    num_workers: int, default=1
        Number of batch generator jobs to run in parallel. If value is set
        to -1 number of machine cores are used.
    """
    assert model_ver in ('dm', 'dbow')
    assert vec_combine_method in ('sum', 'concat')

    dataset = load_dataset(data_file_name)
    nce_data = NCEData(
        dataset,
        batch_size,
        context_size,
        num_noise_words,
        max_generated_batches,
        num_workers)
    nce_data.start()

    try:
        _run(data_file_name, dataset, nce_data.get_generator(), len(nce_data),
             nce_data.vocabulary_size(), context_size, num_noise_words, vec_dim,
             num_epochs, batch_size, lr, model_ver, vec_combine_method,
             save_all)
    except KeyboardInterrupt:
        nce_data.stop()


def _run(data_file_name,
         dataset,
         data_generator,
         num_batches,
         vocabulary_size,
         context_size,
         num_noise_words,
         vec_dim,
         num_epochs,
         batch_size,
         lr,
         model_ver,
         vec_combine_method,
         save_all):

    model = DistributedMemory(
        vec_dim,
        num_docs=len(dataset),
        num_words=vocabulary_size)

    cost_func = NegativeSampling()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

    if torch.cuda.is_available():
        model.cuda()

    print("Dataset comprised of {:d} documents.".format(len(dataset)))
    print("Vocabulary size is {:d}.\n".format(vocabulary_size))
    print("Training started.")

    best_loss = float_info.max
    prev_model_file_path = ""

    for epoch_i in range(num_epochs):
        epoch_start_time = time.time()
        loss = []

        for batch_i in range(num_batches):
            batch = next(data_generator)
            x = model.forward(
                batch.context_ids,
                batch.doc_ids,
                batch.target_noise_ids)
            x = cost_func.forward(x)
            loss.append(x.data[0])
            model.zero_grad()
            x.backward()
            optimizer.step()
            _print_progress(epoch_i, batch_i, num_batches)

        # end of epoch
        loss = torch.mean(torch.FloatTensor(loss))
        is_best_loss = loss < best_loss
        best_loss = min(loss, best_loss)

        model_file_name = MODEL_NAME.format(
            data_file_name[:-4],
            model_ver,
            vec_combine_method,
            context_size,
            num_noise_words,
            vec_dim,
            batch_size,
            lr,
            epoch_i + 1,
            loss)
        model_file_path = join(MODELS_DIR, model_file_name)
        state = {
            'epoch': epoch_i + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer_state_dict': optimizer.state_dict()
        }
        if save_all:
            torch.save(state, model_file_path)
        elif is_best_loss:
            try:
                remove(prev_model_file_path)
            except FileNotFoundError:
                pass
            torch.save(state, model_file_path)
            prev_model_file_path = model_file_path

        epoch_total_time = round(time.time() - epoch_start_time)
        print(" ({:d}s) - loss: {:.4f}".format(epoch_total_time, loss))


def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1), end='')
    stdout.write(" - {:d}%".format(progress))
    stdout.flush()


if __name__ == '__main__':
    fire.Fire()
