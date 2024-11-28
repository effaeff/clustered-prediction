"""Neural network trainer class for time series modeling"""

import os
import re

import numpy as np
import math

from pytorchutils.globals import torch, DEVICE, nn
from pytorchutils.basic_trainer import BasicTrainer
from pytorchutils.cnn import CNNModel
import misc

from sklearn.metrics import mean_squared_error

from config import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    N_CLUSTER,
    MODEL_DIR,
    N_WINDOW,
    BATCH_SIZE,
    PLOT_DIR,
    RESULTS_DIR
)

from tqdm import tqdm

class ClusterTrainer:
    """Wrapper for multiple trainers for each cluster"""
    def __init__(self, config, dataprocessor):
        self.models = np.array([nn.DataParallel(CNNModel(config)) for __ in range(N_CLUSTER)])
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

    def validate(self, save_eval, save_suffix=''):
        __, test_scenarios = self.dataprocessor.get_train_test()
        test_scenarios = self.dataprocessor.window_scenarios(test_scenarios)

        errors = np.zeros(OUTPUT_SIZE)
        variances = np.zeros(OUTPUT_SIZE)

        for scenario_idx, test_scenario in enumerate(tqdm(test_scenarios)):

            exp_number = int(test_scenario[0, -1, INPUT_SIZE+OUTPUT_SIZE])

            total_inp = np.array([])
            pred = np.array([])
            out = np.array([])

            plot_dir = f'{PLOT_DIR}/CNN'
            results_dir = f'{RESULTS_DIR}/CNN'

            misc.gen_dirs([plot_dir, results_dir])

            for cluster_idx in range(N_CLUSTER):
                self.config['models_dir'] = f'{MODEL_DIR}/cluster_{cluster_idx:02d}'

                epoch_dirs = os.listdir(f'{RESULTS_DIR}/CNN/cluster_{cluster_idx:02d}')
                errors = np.empty(len(epoch_dirs))
                for epoch_idx, epoch_dir in enumerate(epoch_dirs):
                    res = np.genfromtxt(
                        f'{RESULTS_DIR}/CNN/cluster_{cluster_idx:02d}/{epoch_dir}/results.txt',
                        skip_header=1,
                        usecols=[1, 3, 4, 6]
                    )
                    errors[epoch_idx] = np.mean(res)

                best_epoch = int(re.search(r'\d+', epoch_dirs[np.argmin(errors)]).group())

                trainer = Trainer(self.config, self.models[cluster_idx], self.dataprocessor, best_epoch)

                clustered_scenario = test_scenario[test_scenario[:, -1, -1]==cluster_idx, :, :-1]

                if len(clustered_scenario) > BATCH_SIZE:
                    clustered_scenario = clustered_scenario[
                        :len(clustered_scenario) // BATCH_SIZE * BATCH_SIZE
                    ]
                elif len(clustered_scenario)==0:
                    continue

                clustered_inp = clustered_scenario[:, :, :INPUT_SIZE]
                time = clustered_scenario[:, -1, 0]

                cluster_pred = np.empty((len(clustered_inp), OUTPUT_SIZE))

                inp = np.reshape(
                    clustered_inp,
                    (
                        -1,
                        BATCH_SIZE if len(clustered_scenario) >= BATCH_SIZE else len(clustered_scenario),
                        1,
                        N_WINDOW,
                        clustered_inp.shape[-1]
                    )
                )
                cluster_out = clustered_scenario[:, -1, INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE]

                trainer.model.eval()
                inp = torch.Tensor(inp).to(DEVICE)

                if len(inp) == 1:
                    batch_pred = trainer.model(inp[0])
                    batch_pred = torch.sigmoid(batch_pred)
                    cluster_pred = batch_pred.cpu().detach().numpy()
                else:
                    for batch_idx, batch in enumerate(inp):
                        batch_pred = trainer.model(batch)
                        batch_pred = torch.sigmoid(batch_pred)
                        cluster_pred[
                            batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE + BATCH_SIZE
                        ] = batch_pred.cpu().detach().numpy()

                timed_pred = np.c_[time, cluster_pred]
                timed_out = np.c_[time, cluster_out]

                total_inp = np.vstack(
                    [total_inp, clustered_inp[:, -1, :]]
                ) if total_inp.size else clustered_inp[:, -1, :]
                pred = np.vstack([pred, timed_pred]) if pred.size else timed_pred
                out = np.vstack([out, timed_out]) if out.size else timed_out

            total_inp = total_inp[total_inp[:, 0].argsort(), :]
            pred = pred[pred[:, 0].argsort(), 1:]
            out = out[out[:, 0].argsort(), 1:]

            for out_idx in range(OUTPUT_SIZE):
                errors[out_idx] = math.sqrt(
                    mean_squared_error(out[:, out_idx], pred[:, out_idx])
                ) / np.ptp(out[:, out_idx]) * 100.0
                variances[out_idx] = np.std(
                    [
                        abs(out[idx, out_idx] - pred[idx, out_idx]) / np.ptp(out[:, out_idx])
                        for idx in range(len(out))
                    ]
                ) * 100.0

            self.dataprocessor.plot_validation_scenario(
                total_inp,
                pred,
                out,
                exp_number,
                plot_dir,
                scenario_idx,
                save_eval,
                save_suffix
            )
            if save_eval:
                np.savez(
                    f'{results_dir}/CNN_scenario{scenario_idx}_expno{exp_number}{save_suffix}.npz',
                    pred=pred,
                    target=out
                )
        if save_eval:
            with open(
                f'{results_dir}/results.txt',
                'w',
                encoding='utf-8'
            ) as res_file:
                res_file.write(
                    f"{'Regressor':<40} {'NRMSE dx':<40} NRMSE dy\n"
                )
                res_file.write(
                    f"{'CNN':<40} "
                    f"{f'{errors[0]:.2f} +/- {variances[0]:.2f}':<40} "
                    f"{f'{errors[1]:.2f} +/- {variances[1]:.2f}'}\n"
                )
        print(f'Validation error: {np.mean(errors)} +- {np.mean(variances)}')


class Trainer(BasicTrainer):
    """Wrapper class for training routine"""
    def __init__(self, config, model, dataprocessor, load_epoch):
        BasicTrainer.__init__(self, config, model, dataprocessor, load_epoch)

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
