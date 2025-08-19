# Import necessary libraries
import tqdm 
import warnings

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS, TSNE 
from umap import UMAP
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist,squareform


class DimensionReducer():
    """
    Class for reducing the dimensionality of a dataset using various techniques.
    """
    def __init__(self, X, labels):
        """
        Initialize the DimensionReducer with a dataset and labels.
        The dataset is scaled to [0, 1] range.
        """
        self.X = MinMaxScaler().fit_transform(X)
        self.labels = labels
        self.D = squareform(pdist(self.X))

    def compute_MDS(self):
        """
        Compute MDS (Multidimensional Scaling) on the dataset.
        """
        Y = MDS(dissimilarity="precomputed").fit_transform(self.D)
        return Y

    def compute_TSNE(self):
        """
        Compute t-SNE (t-Distributed Stochastic Neighbor Embedding) on the dataset.
        """
        Y = TSNE(metric='precomputed', init='random').fit_transform(self.D)
        return Y

    def compute_UMAP(self):
        """
        Compute UMAP (Uniform Manifold Approximation and Projection) on the dataset.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            Y = UMAP(metric='precomputed',init='random').fit_transform(self.D)
            return Y

    def compute_random(self):
        """
        Generate a random 2D embedding of the dataset.
        """
        Y = np.random.uniform(0, 1, (self.X.shape[0], 2))
        return Y
    
    def compute_Isomap(self):
        """
        Compute Isomap embedding on the dataset.
        """
        Y = Isomap(n_components=2, metric='precomputed', n_neighbors=15).fit_transform(self.D)
        return Y

    def compute_LLE(self):
        """
        Compute Locally Linear Embedding (LLE) on the dataset.
        precomputed distance matrix is not used
        LLE only works with Euclidean distances
        """
        Y = LocallyLinearEmbedding(n_components=2).fit_transform(self.X)
        return Y

    def compute_PCA(self):
        """
        Compute PCA (Principal Component Analysis) on the dataset.
        """
        Y = PCA(n_components=2).fit_transform(self.X)
        return Y

def save_embeddings(data, name, i, folder):
    """
    Save the embeddings computed by various techniques into a specified folder.
    """
    DR = DimensionReducer(*data)
    computations = ['MDS', 'TSNE', 'UMAP', 'random']
    methods = [DR.compute_MDS, DR.compute_TSNE, DR.compute_UMAP, DR.compute_random]

    for comp, method in zip(computations, methods):
        result = method()
        np.save(f"{folder}/{name}_{i}_{comp.lower()}.npy", result)
        if folder == 'big_data_embeddings':
            np.save(f"{folder}/{name}_{i}.npy", data[0])


if __name__ == "__main__":
    """
    Main function to compute and save embeddings for all datasets in the 'datasets' directory.
    """
    EMBEDDINGS_FOLDER = "embeddings"


    import os
    if not os.path.isdir(EMBEDDINGS_FOLDER):
        os.mkdir(EMBEDDINGS_FOLDER)

    datasets = os.listdir('datasets')
    # datasets = ['epileptic.npy']
    
    num_iter = 10
    num_algs = 7
    with tqdm.tqdm(total=len(datasets) * num_iter * num_algs) as pbar:

        for datasetStr in datasets:
            X = np.load(f"datasets/{datasetStr}")

            DR = DimensionReducer(X,None)
            dname = datasetStr.replace(".npy", "")

            for i in range(num_iter):
                random = DR.compute_random()
                np.save(f"{EMBEDDINGS_FOLDER}/{dname}_RANDOM_{i}.npy",random)                                    
                pbar.update(1)

                umap = DR.compute_UMAP()
                np.save(f"{EMBEDDINGS_FOLDER}/{dname}_UMAP_{i}.npy",umap)
                pbar.update(1)

                tsne = DR.compute_TSNE()
                np.save(f"{EMBEDDINGS_FOLDER}/{dname}_TSNE_{i}.npy",tsne)
                pbar.update(1)

                mds = DR.compute_MDS()
                np.save(f"{EMBEDDINGS_FOLDER}/{dname}_MDS_{i}.npy",mds)
                pbar.update(1)

                lle = DR.compute_LLE()
                np.save(f"{EMBEDDINGS_FOLDER}/{dname}_LLE_{i}.npy",lle)                                    
                pbar.update(1)

                pca = DR.compute_PCA()
                np.save(f"{EMBEDDINGS_FOLDER}/{dname}_PCA_{i}.npy",pca)
                pbar.update(1)

                isomap = DR.compute_Isomap()
                np.save(f"{EMBEDDINGS_FOLDER}/{dname}_Isomap_{i}.npy",isomap)
                pbar.update(1)