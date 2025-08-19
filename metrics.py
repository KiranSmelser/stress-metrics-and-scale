# Import necessary libraries
import numpy as np 
from sklearn.metrics import pairwise_distances
import zadu
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar

MACHINE_EPSILON = np.finfo(np.float64).eps

class Metrics():
    """
    Class for computing various stress metrics between high-dimensional and low-dimensional data.
    """
    def __init__(self, X, Y, 
            setbatch=True, precomputed=False, scaling_factors = None):
        """
        Initialize the Metrics class with high-dimensional data X and low-dimensional data Y.
        Compute pairwise distances within X and Y.
        Also compute pairwise distances within Y for different scales in scaling_factors

        Parameters
        ----------
        X : 2D array
          Dataset as numpy array

        Y : 2D array
          Embedding as numpy array

        setbatch: bool
          If True, computes embeddings at the different scales specified in scaling_factors

        precomputed: bool
          If True, X and Y are considered to be the distance matrices dX and dY of the dataset and embedding respectively.
          All computations that explicitly require the original data will either return an error or not work as expected.

        scaling_factors: 1D array
          Scales at which to rescale Y

        """
        if precomputed:
            self.dX = X
            self.dY = Y
        else:
            self.X = X
            self.Y = Y 

            self.dX = pairwise_distances(X)
            self.dY = pairwise_distances(Y)

        if setbatch:
            self.setBatch(scaling_factors, precomputed=precomputed)

    def setBatch(self,scaling_factors=np.linspace(MACHINE_EPSILON,20,100), precomputed=False):
        if precomputed:
            self.dY_batch = self.Y[None, :, :] * scaling_factors[:, None, None]
        else:
            self.Y_batch = self.Y[None , :, :] * scaling_factors[ :, None, None]
            self.dY_batch = np.array([pairwise_distances(Y) for Y in self.Y_batch])     

    def setY(self,Y):
        """
        Update low-dimensional data Y and compute pairwise distances within Y.
        """
        self.Y = Y 
        self.dY = pairwise_distances(Y)

    def compute_raw_stress(self):
        """
        Compute raw stress between pairwise distances of X and Y.
        """
        return np.sum(np.square(self.dX - self.dY)) / 2

    def compute_normalized_stress(self,alpha=1.0):  
        """
        Compute normalized stress between X and alpha*Y using zadu's stress measure.
        """      
        from zadu.measures import stress
        stressScore = stress.measure(self.X,alpha * self.Y,(self.dX, alpha * self.dY))
        return stressScore['stress']

    def compute_fs_normalized_stress(self):
        """
        Compute Forced Scale Normalized Stress (Normalize dataset and embeddings before
        computing normalized stress.)
        """
        dX = self.dX / np.max(self.dX)
        dY = self.dY / np.max(self.dY)
        
        numerator = np.sum(np.square(dX - dY))
        denominator = np.sum(np.square(dX))
        return np.sqrt(numerator / denominator)


    def compute_scale_normalized_stress(self,return_alpha=False):
        """
        Compute scale-normalized stress between pairwise distances of X and Y.
        Optimal scaling factor alpha is computed as well.
        """
        D_low_triu = self.dY[np.triu_indices(self.dY.shape[0], k=1)]
        D_high_triu = self.dX[np.triu_indices(self.dX.shape[0], k=1)]
        alpha = np.sum(D_low_triu * D_high_triu) / np.sum(np.square(D_low_triu))
        if return_alpha:
            return self.compute_normalized_stress(alpha), alpha
        return self.compute_normalized_stress(alpha)

    def compute_kruskal_stress(self):
        """
        Compute Kruskal's non-metric stress between pairwise distances of X and Y. Invariant to scale of Y.
        """

        dij = self.dX[np.triu_indices(self.dX.shape[0], k=1)]
        xij = self.dY[np.triu_indices(self.dY.shape[0], k=1)]

        # Find the indices of dij that when reordered, would sort it. Apply to both arrays
        sorted_indices = np.argsort(dij)
        dij = dij[sorted_indices]
        xij = xij[sorted_indices]

        from sklearn.isotonic import IsotonicRegression
        hij = IsotonicRegression().fit(dij, xij).predict(dij)

        raw_stress = np.sum(np.square(xij - hij))
        norm_factor = np.sum(np.square(xij))

        kruskal_stress = np.sqrt(raw_stress / norm_factor)
        return kruskal_stress

    def compute_shepard_correlation(self):
        """
        Compute Shepard's correlation between pairwise distances of X and Y using zadu's spearman_rho measure.
        Invariant to scale of Y.
        """
        from zadu.measures import spearman_rho
        shepardCorr = spearman_rho.measure(self.X,self.Y,(self.dX,self.dY))
        return shepardCorr['spearman_rho']
    
    def compute_kl_divergence(self, perplexity=30, y_similarity='t'):
        """
        Compute joint probability distributions of X and Y à la t-SNE and compute the KL divergence between them

        Parameters
        ----------
        perplexity : Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf
        """
        # High-dimensional probability space
        P = self._joint_probabilities(perplexity)

        # Low-dimensional probability space
        Q = self._get_Q(is_batch=False, similarity=y_similarity)

        # KL Divergence
        kl_divergence = (P * np.log(P / Q)).sum()
        return kl_divergence
    
    def compute_kl_divergences(self, perplexity, y_similarity='t'):
        """
        Compute joint probability distributions of X and Y à la t-SNE and compute the KL divergence between them

        Parameters
        ----------
        perplexity : float
            Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf
        """

        # High-dimensional probability space
        P = self._joint_probabilities(perplexity)

        # Low-dimensional probability space for all Y_batch
        Q_batch = self._get_Q(is_batch=True, similarity=y_similarity)

        # KL Divergence
        kl_divergences = (P * np.log(P / Q_batch)).sum(axis=(1, 2))
        return kl_divergences

    def SNKL(self, perplexity=30, ret_scale=True, max_bound=300):
        """
        Calculate Scale-Normalized KL Divergence (KL divergence and scale at the scale KL divergence is minimum)

        Parameters
        ----------

        perplexity : float
            Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf

        ret_scale : bool
            if True, returns scale, min_KL tuple
            else, returns min_KL

        max_bound : positive number
            upper bound of interval within which minimum is searched (If returned KL divergence is almost equal to max_bound, max_bound was likely too small)
        """
        # High-dimensional probability space
        P = self._joint_probabilities(perplexity)

        # Calculate KL divergence for scale=scale
        def get_kl(scale):
            Q = self._get_Q(is_batch=False, scale=scale, similarity='t')
            kl_divergence = (P * np.log(P / Q)).sum()
            return kl_divergence
        
        # Calculate minimum KL
        res = minimize_scalar(get_kl, bounds=(0, max_bound))

        if ret_scale:
            return (res.x, res.fun)
        else:
            return res.fun

    def compute_kl_divergence_at_infty(self, perplexity):
        """
        Compute the KL Divergence between probability distributions of X and Y à la t-SNE at infinite scale of the embedding

        Parameters
        ----------
        perplexity : float
            Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf
        """

        # High-dimensional probability space
        P = self._joint_probabilities(perplexity)

        # Low dimensional probability space
        Q = np.maximum(self.dY, MACHINE_EPSILON) ** -2
        np.fill_diagonal(Q, 0)
        Q /= Q.sum()
        Q = np.maximum(Q, 1e-15)

        # KL Divergence
        kl_divergence = (P * np.log(P / Q)).sum()

        return kl_divergence
    
    def _get_Q(self, is_batch: bool, scale: int = 1, similarity='t'):
        """
        Calculate the low dimensional probability distribution corresponding to Y
        Parameters
        ----------
        is_batch : bool
          If True, calculates probability distribution for all scales of embedding
          If False, only calculates for scale = 1
        
        similarity : str
          't' -> The Student's t-distribution is used (like in t-SNE)
          'normal' -> The normal distribution is used (like in SNE)

        scale : float
          default = 1
          Only relevant if is_batch = False and similarity = 't'.
          Scale to scale self.dY by
        """
        # # Fill Q
        # if similarity == 't':
        #     if is_batch:
        #         Q = (np.square(self.dY_batch) + 1.0) ** -1
        #     else:
        #         Q = (np.square(self.dY) + 1.0) ** -1
        # elif similarity == 'normal':
        #     if is_batch:
        #         Q = np.exp(-1 * np.square(self.dY_batch))
        #     else:
        #         Q = np.exp(-1 * np.square(self.dY))
        #     Q = np.maximum(Q, MACHINE_EPSILON)
        # else:
        #     raise ValueError(f"Invalid string: '{similarity}' is not a valid value for the parameter 'similarity'")

        # # Fill diagonal with 0
        # if is_batch:
        #     rows = np.arange(Q.shape[1])
        #     Q[:, rows, rows] = 0 # Fill diagonal with 0
        #     Q /= Q.sum(axis=(1, 2), keepdims=True)
        # else:
        #     np.fill_diagonal(Q, 0)
        #     Q /= Q.sum()

        # Q = np.maximum(Q, 1e-15)

        if similarity == 't':
            if is_batch:
                Q = (np.square(self.dY_batch) + 1.0) ** -1
                rows = np.arange(Q.shape[1])
                Q[:, rows, rows] = 0 # Fill diagonal with 0
                Q /= Q.sum(axis=(1, 2), keepdims=True)
            else:
                Q = (np.square(self.dY * scale) + 1.0) ** -1
                np.fill_diagonal(Q, 0)
                Q /= Q.sum()
        
        elif similarity == 'normal':
            if is_batch:
                negsq_dist = -np.square(self.dY_batch)
                rows = np.arange(negsq_dist.shape[1])
                negsq_dist[:, rows, rows] = -np.inf # Fill diagonal with -inf (since exp(-inf) == 0)
                Q = np.exp(negsq_dist - logsumexp(negsq_dist, axis=(1,2), keepdims=True))
            else:
                negsq_dist = -np.square(self.dY)
                np.fill_diagonal(negsq_dist, -np.inf) # Fill diagonal with -inf (since exp(-inf) == 0)
                Q = np.exp(negsq_dist - logsumexp(negsq_dist, keepdims=True))
        else:
            raise ValueError(f"Invalid string: '{similarity}' is not a valid value for the parameter 'similarity'")
            
        # To prevent zero division error
        Q = np.maximum(Q, 1e-15)

        return Q

    def _joint_probabilities(self, perplexity):
        n_samples = self.dX.shape[0]
        conditional_P = self._conditional_probabilities(perplexity)
        P = (conditional_P + conditional_P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-15)
        return P

    def _conditional_probabilities(self, perplexity, steps=100):
        """
        Calculate conditional probability matrix P corresponding to SNE algorithm
        
        P[i, j] = P (j | i)
        
        Parameters
        ----------
        perplexity : Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf
        steps : Number of steps for binary search for variances of Gaussian distributions
        """
        n_samples = self.dX.shape[0]
        desired_entropy = np.full((n_samples, 1), np.log(perplexity))
        beta = np.ones((n_samples, 1)) # (2 * var_i ** 2) in the exponent of the Gaussian distribution
        beta_min = np.zeros((n_samples, 1), dtype=np.float64)
        beta_max = np.full((n_samples, 1), np.inf)

        # Binary Search
        for _ in range(steps):
            # Create conditional probability distributions
            P = np.exp(-1 * np.square(self.dX) / beta)
            np.fill_diagonal(P, 0)
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-8
            P = P / row_sums
            P = np.maximum(P, 1e-15)

            # Calculate entropy
            log_P = np.log(P)
            entropy = -1 * (P * log_P).sum(axis=1, keepdims=True)
            entropy_diff = entropy - desired_entropy


            # Stop if variances sufficiently match perplexity
            if np.all(np.abs(entropy_diff) < 1e-5):
                break
            

            # Binary search update:

            should_increase_beta = entropy_diff < 0.0

            # Update beta_min and beta_max
            beta_min = np.where(should_increase_beta, beta, beta_min)
            beta_max = np.where(should_increase_beta, beta_max, beta)

            beta_max_is_inf = (beta_max == np.inf)
        
            # Update beta

            mask = should_increase_beta & beta_max_is_inf
            beta[mask] = beta[mask] * 2

            mask = should_increase_beta & ~beta_max_is_inf
            beta[mask] = (beta[mask] + beta_max[mask]) / 2.0

            mask = ~should_increase_beta
            beta[mask] = (beta[mask] + beta_min[mask]) / 2.0

        return P


if __name__ == "__main__":
    """
    Main function to compute and plot normalized stress for a range of scaling factors.
    """
    X = np.load('datasets/auto-mpg.npy')
    Y = np.load('embeddings/auto-mpg-TSNE-0.npy')

    M = Metrics(X,Y)
    
    scale_opt, alpha_opt = M.compute_scale_normalized_stress(return_alpha=True)

    # Compute normalized stress for a range of scaling factors
    rrange = np.linspace(0,100,2000)
    norm_stress_scores = list()
    for alpha in rrange:
        M.setY(alpha * Y)
        norm_stress_scores.append(M.compute_normalized_stress())

    # Plot normalized stress scores and highlight the optimal scaling factor
    import pylab as plt 
    plt.plot(rrange, norm_stress_scores)
    plt.scatter(alpha_opt, scale_opt)
    plt.show()