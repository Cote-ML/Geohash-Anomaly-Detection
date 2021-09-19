import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.utils import logger

LOGGER = logger.setup_logger(__name__)


def pca_cumulative(df, epsilon=1):
    cum_var = np.cumsum(
        np.round(PCA(n_components=min(df.shape)).fit(df).explained_variance_ratio_, decimals=4) * 100)
    cum_var_diff = np.array([round(j - i, 2) for i, j in zip(cum_var[:-1], cum_var[1:])])
    cutoff = np.argmax(cum_var_diff < epsilon) + 1
    LOGGER.debug("Number Components for Convergence of PCA Variance: {}".format(cutoff))
    pca_df = pd.DataFrame(PCA(n_components=cutoff).fit_transform(df))
    return pca_df