"""Clustering of time series data"""

import os
import numpy as np

from matplotlib import pyplot as plt
plt.rc('axes', axisbelow=True)
from matplotlib.patches import Rectangle
from matplotlib import colormaps as cm

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score

from plot_utils import modify_axis, CM_INCH
from colors import dark2

from joblib import dump, load

from tqdm import tqdm

from config import (
    DATA_DIR,
    RANDOM_SEED,
    PARAM_FILE,
    MODEL_DIR,
    FONTSIZE,
    PLOT_DIR,
    RESULTS_DIR,
    N_CLUSTER,
    N_CLUSTER_SILH,
    CLUSTER_COLS,
    VERBOSE
)

class Clusterer:
    """Data processing class"""
    def __init__(self):
        np.random.seed(RANDOM_SEED)

    def cluster(self, train_data, test_data, cluster_cols):
        """
        Data has columns: time, Fx, Fy, Fz, path_x, path_y, spsp, fz, r1, r2, dx, dy.
        Clusters train and test data using cluster_cols as features.
        Add additional column to each data array indicating the cluster index.
        """
        # clusterer_fname = f'{MODEL_DIR}/kmeans.joblib'
        clusterer_fname = f'{MODEL_DIR}/gmm'

        if os.path.isfile(clusterer_fname):
            print(f'Loading clusterer {clusterer_fname}')
            # clusterer = load(clusterer_fname)

            means = np.load(f'{clusterer_fname}_means.npy')

            clusterer = GaussianMixture(n_components = len(means), covariance_type='full')

            clusterer.means_ = means
            clusterer.precisions_cholesky_ = np.load(f'{clusterer_fname}_precisions_cholesky.npy')
            clusterer.weights_ = np.load(f'{clusterer_fname}_weights.npy')
            clusterer.covariances_ = np.load(f'{clusterer_fname}_covariances.npy')
        else:
            # clusterer = KMeans(n_clusters=N_CLUSTER, random_state=RANDOM_SEED, n_init=10)
            clusterer = GaussianMixture(
                n_components=N_CLUSTER,
                covariance_type='full',
                n_init=10,
                random_state=RANDOM_SEED
            )
            print(f'Fitting clusterer {clusterer_fname}')
            clusterer.fit(train_data[:, cluster_cols])

            np.save(f'{clusterer_fname}_weights.npy', clusterer.weights_, allow_pickle=False)
            np.save(f'{clusterer_fname}_means.npy', clusterer.means_, allow_pickle=False)
            np.save(f'{clusterer_fname}_covariances.npy', clusterer.covariances_, allow_pickle=False)
            np.save(
                f'{clusterer_fname}_precisions_cholesky.npy',
                clusterer.precisions_cholesky_,
                allow_pickle=False
            )

            # dump(
                # clusterer,
                # clusterer_fname
            # )

        cluster_labels = clusterer.predict(train_data[:, cluster_cols])
        train_data = np.c_[train_data, cluster_labels]

        for test_idx, test_scenario in enumerate(test_data):
            cluster_labels = clusterer.predict(test_scenario[:, cluster_cols])
            test_data[test_idx] = np.c_[test_scenario, cluster_labels]
            # self.plot_cluster(np.c_[test_scenario, cluster_labels])

        # self.plot_cluster(train_data)

        return train_data, test_data

    def plot_cluster(self, data):
        labels = data[:, -1]
        colors = cm.get_cmap('Spectral')(labels.astype(float) / N_CLUSTER)

        fig, axs = plt.subplots(1, 2, figsize=(30*CM_INCH, 10*CM_INCH), layout='constrained')

        axs[0].add_patch(
            Rectangle((-34.5, -34.5), 69, 69, edgecolor='#1b9e77', facecolor='none', label='Workpiece')
        )

        # Path data are indices 4 and 5
        axs[0].scatter(data[:, 4], data[:, 5], c=colors, s=3)

        axs[0].set_xlabel("x-coordinate", fontsize=FONTSIZE)
        axs[0].set_ylabel("y-coordinate", fontsize=FONTSIZE)

        axs[0].set_xticks(np.arange(-45, 46, 30))
        axs[0].set_yticks(np.arange(-45, 46, 30))

        # Use Fy for force visualization and don't use time column to avoid overlap in the plot
        axs[1].scatter([idx for idx in range(len(data))], data[:, 2], c=colors, s=1)
        axs[1].set_xlabel("Sample", fontsize=FONTSIZE)
        axs[1].set_ylabel("Force", fontsize=FONTSIZE)

        # axs_cluster[1].set_yticks(np.arange(-50, 151, 100))
        # axs_cluster[1].set_xticks(np.arange(0, 4, 1))

        # fig.canvas.draw()

        # axs_cluster[0] = modify_axis(axs_cluster[0], 'mm', 'mm', -2, -2, FONTSIZE, grid=False)
        # axs_cluster[1] = modify_axis(axs_cluster[1], 's', 'N', -2, -2, FONTSIZE, grid=True)
        # axs_cluster[2] = modify_axis(axs_cluster[2], 's', 'N', -2, -2, FONTSIZE, grid=True)

        axs[0].set_xlim(-45, 45)
        axs[0].set_ylim(-45, 45)

        # fig_cluster.savefig(f'{PLOT_DIR}/{exp_number}/gmm_selection_slides.png', dpi=600, transparent=True)
        # fig_cluster.savefig(f'{PLOT_DIR}/{exp_number}/kmeans_all_slides.png', dpi=600, transparent=True)

        plt.show()

    def silhouette_analysis(self, data):
        """Propably not working right now"""
        for n_clusters in tqdm(N_CLUSTER_SILH):
            fig_sil, axs_fil = plt.subplots(1, 3, figsize=(30*CM_INCH, 5*CM_INCH), layout='constrained')
            ax1 = axs_fil[0]
            ax2 = axs_fil[1]
            ax3 = axs_fil[2]

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init=10)
            # clusterer = BayesianGaussianMixture(
                # n_components=n_clusters,
                # covariance_type='full',
                # n_init=10,
                # init_params='k-means++',
                # random_state=10
            # )
            # clusterer = load(f'{RESULTS_DIR}/{exp_number}/{n_clusters}_clusterer.joblib')
            cluster_labels = clusterer.fit_predict(data)
            # cluster_labels = clusterer.predict(features)
            # dump(clusterer, f'{RESULTS_DIR}/{exp_number}/{n_clusters}_clusterer.joblib')

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(data, cluster_labels)
            # np.save(f'{RESULTS_DIR}/{exp_number}/{n_clusters}_sil-avg.npy', silhouette_avg)
            # silhouette_avg = np.load(f'{RESULTS_DIR}/{exp_number}/{n_clusters}_sil-avg.npy')
            # print(
                # "For n_clusters =",
                # n_clusters,
                # "The average silhouette_score is :",
                # silhouette_avg,
            # )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(data, cluster_labels)
            # np.save(f'{RESULTS_DIR}/{exp_number}/{n_clusters}_sil-scores.npy', sample_silhouette_values)
            # sample_silhouette_values = np.load(f'{RESULTS_DIR}/{exp_number}/{n_clusters}_sil-scores.npy')

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.get_cmap('Spectral')(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                    rasterized=True
                )

                # Label the silhouette plots with their cluster numbers at the middle
                # ax1.text(-0.1, y_lower + 0.5 * size_cluster_i, str(i), fontsize=FONTSIZE)

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("Silhouette coefficient value", fontsize=FONTSIZE)
            ax1.set_ylabel("Cluster", fontsize=FONTSIZE)

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.get_cmap('Spectral')(cluster_labels.astype(float) / n_clusters)

            ax2.add_patch(
                Rectangle((-34.5, -34.5), 69, 69, edgecolor='#1b9e77', facecolor='none', label='Workpiece')
            )

            ax2.scatter(data[:, 0], data[:, 1], c=colors, s=3)

            # Labeling the clusters
            # centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            # ax2.scatter(
                # centers[:, 0],
                # centers[:, 1],
                # marker="o",
                # c="white",
                # alpha=1,
                # s=200,
                # edgecolor="k",
            # )

            # for i, c in enumerate(centers):
                # ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            # ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("x-coordinate", fontsize=FONTSIZE)
            ax2.set_ylabel("y-coordinate", fontsize=FONTSIZE)

            ax3.set_xlabel("Time", fontsize=FONTSIZE)
            ax3.set_ylabel("Force", fontsize=FONTSIZE)

            # fig_sil.suptitle(
                # "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                # % n_clusters,
                # fontsize=14,
                # fontweight="bold",
            # )

            # fig_sil.tight_layout()


