from os import remove
from os.path import join, dirname, isfile

import matplotlib.pyplot as plt
import torch

_root_dir = dirname(dirname(__file__))

DATA_DIR = join(_root_dir, 'data')
MODELS_DIR = join(_root_dir, 'models')
_DIAGNOSTICS_DIR = join(_root_dir, 'diagnostics')

_DM_MODEL_NAME = ("{:s}_model.{:s}.{:s}_contextsize.{:d}_numnoisewords.{:d}"
                  "_vecdim.{:d}_batchsize.{:d}_lr.{:f}_epoch.{:d}_loss.{:f}"
                  ".pth.tar")
_DM_DIAGNOSTIC_FILE_NAME = ("{:s}_model.{:s}.{:s}_contextsize.{:d}"
                            "_numnoisewords.{:d}_vecdim.{:d}_batchsize.{:d}"
                            "_lr.{:f}.csv")
_DBOW_MODEL_NAME = ("{:s}_model.{:s}_numnoisewords.{:d}_vecdim.{:d}"
                    "_batchsize.{:d}_lr.{:f}_epoch.{:d}_loss.{:f}.pth.tar")
_DBOW_DIAGNOSTIC_FILE_NAME = ("{:s}_model.{:s}_numnoisewords.{:d}_vecdim.{:d}"
                              "_batchsize.{:d}_lr.{:f}.csv")


def save_training_state(data_file_name,
                        model_ver,
                        vec_combine_method,
                        context_size,
                        num_noise_words,
                        vec_dim,
                        batch_size,
                        lr,
                        epoch_i,
                        loss,
                        model_state,
                        save_all,
                        generate_plot,
                        is_best_loss,
                        prev_model_file_path,
                        model_ver_is_dbow):
    """Saves the state of the model. If generate_plot is True, it also
    saves current epoch's loss value and generates a plot of all loss
    values up to this epoch.

    Returns
    -------
        str representing a model file path from the previous epoch
    """
    if generate_plot:
        # save the loss value for a diagnostic plot
        if model_ver_is_dbow:
            diagnostic_file_name = _DBOW_DIAGNOSTIC_FILE_NAME.format(
                data_file_name[:-4],
                model_ver,
                num_noise_words,
                vec_dim,
                batch_size,
                lr)
        else:
            diagnostic_file_name = _DM_DIAGNOSTIC_FILE_NAME.format(
                data_file_name[:-4],
                model_ver,
                vec_combine_method,
                context_size,
                num_noise_words,
                vec_dim,
                batch_size,
                lr)

        diagnostic_file_path = join(_DIAGNOSTICS_DIR, diagnostic_file_name)

        if epoch_i == 0 and isfile(diagnostic_file_path):
            remove(diagnostic_file_path)

        with open(diagnostic_file_path, 'a') as f:
            f.write('{:f}\n'.format(loss))

        # generate a diagnostic loss plot
        with open(diagnostic_file_path) as f:
            loss_values = [float(l.rstrip()) for l in f.readlines()]

        diagnostic_plot_file_path = diagnostic_file_path[:-3] + 'png'
        fig = plt.figure()
        plt.plot(range(1, epoch_i + 2), loss_values, color='r')
        plt.xlabel('epoch')
        plt.ylabel('training loss')
        fig.savefig(diagnostic_plot_file_path, bbox_inches='tight')
        plt.close()

    # save the model
    if model_ver_is_dbow:
        model_file_name = _DBOW_MODEL_NAME.format(
            data_file_name[:-4],
            model_ver,
            num_noise_words,
            vec_dim,
            batch_size,
            lr,
            epoch_i + 1,
            loss)
    else:
        model_file_name = _DM_MODEL_NAME.format(
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

    if save_all:
        torch.save(model_state, model_file_path)
        return None
    elif is_best_loss:
        if prev_model_file_path is not None:
            remove(prev_model_file_path)

        torch.save(model_state, model_file_path)
        return model_file_path
    else:
        return prev_model_file_path
