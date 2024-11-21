"""Neural network trainer class for time series modeling"""

import numpy as np

from pytorchutils.globals import torch, DEVICE
from pytorchutils.basic_trainer import BasicTrainer
from pytorchutils.cnn import CNNModel
import misc

from config import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    N_CLUSTER,
    MODEL_DIR,
    BATCH_SIZE
)

from tqdm import tqdm

class ClusterTrainer:
    """Wrapper for multiple trainers for each cluster"""
    def __init__(self, config, dataprocessor):
        self.models = np.array([CNNModel(config) for __ in range(N_CLUSTER)])
        self.config = config
        self.dataprocessor = dataprocessor

    def train(self, validate_every, save_every, save_eval, verbose):
        for cluster_idx in range(N_CLUSTER):
            self.config['models_dir'] = f'{MODEL_DIR}/cluster_{cluster_idx:02d}'

            misc.gen_dirs([f'{MODEL_DIR}/cluster_{cluster_idx:02d}'])

            trainer = Trainer(self.config, self.models[cluster_idx], self.dataprocessor)

            trainer.get_batches_fn = self.dataprocessor.get_batches

            self.dataprocessor.set_current_cluster(cluster_idx)

            trainer.train(
                validate_every=validate_every,
                save_every=save_every,
                save_eval=save_eval,
                verbose=verbose
            )

class Trainer(BasicTrainer):
    """Wrapper class for training routine"""
    def __init__(self, config, model, dataprocessor):
        BasicTrainer.__init__(self, config, model, dataprocessor)

    def learn_from_epoch(self, epoch_idx, verbose):
        """Training method"""
        epoch_loss = 0
        try:
            batches = self.get_batches_fn()
        except AttributeError:
            print(
                "Error: No nb_batches_fn defined in preprocessor. "
                "This attribute is required by the training routine."
            )

        pbar = tqdm(batches, desc=f'Epoch {epoch_idx}', unit='batch')
        # total_out = np.reshape(batches[:, :, 0, -1, INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE], (-1, OUTPUT_SIZE))
        # total_pred = np.empty((len(total_out), OUTPUT_SIZE))
        for idx, batch in enumerate(pbar):
            inp = batch[:, :, :, :INPUT_SIZE]
            out = batch[:, 0, -1, INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE]

            pred_out = self.predict(inp)

            pred_out = torch.sigmoid(pred_out)

            batch_loss = self.loss(
                pred_out,
                torch.FloatTensor(out).to(DEVICE)
            )

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()

            # total_pred[idx*BATCH_SIZE:idx*BATCH_SIZE + BATCH_SIZE] = pred_out.cpu().detach().numpy()

            pbar.set_postfix(batch_loss=batch_loss.item(), epoch_loss=epoch_loss/(idx+1))
        epoch_loss /= len(batches)

        # import matplotlib.pyplot as plt
        # __, axs = plt.subplots(2, 1, sharex=True)
        # axs[0].plot(total_out[:, 0])
        # axs[0].plot(total_pred[:, 0])
        # axs[1].plot(total_out[:, 1])
        # axs[1].plot(total_pred[:, 1])
        # plt.show()
        # plt.close()

        return epoch_loss

    def predict(self, inp):
        """
        Capsuled prediction method.
        Only single model usage supported for now.
        """
        inp = torch.Tensor(inp).to(DEVICE)
        return self.model(inp)

    def evaluate(self, inp):
        """Prediction and error estimation for given input and output"""
        with torch.no_grad():
            # Switch to PyTorch's evaluation mode.
            # Some layers, which are used for regularization, e.g., dropout or batch norm layers,
            # behave differently, i.e., are turnd off, in evaluation mode
            # to prevent influencing the prediction accuracy.
            self.model.eval()

            pred_out = self.predict(inp)

            pred_out = torch.sigmoid(pred_out)

            pred_out = pred_out.cpu().detach().numpy()
            return pred_out
