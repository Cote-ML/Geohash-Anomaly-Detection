from sklearn.cluster import DBSCAN
from src.model import geohash, transformation
from src.utils import logger

LOGGER = logger.setup_logger(__name__)


class SpatialClustering(object):
    def __init__(self, target, sigma, cv):
        self.target = target
        self.sigma = sigma
        self.cv = cv
        self.sigfig = 4
        self.eps_range = []

    def fit_predict(self, df):
        labels, proportion = self._auto_optimize_model(df)
        # TODO: log_proportions_here
        return labels

    def _auto_optimize_model(self, df):
        def binary_param_search(df, eps_range, min_idx, max_idx):
            if max_idx > min_idx:
                mid_idx = min_idx + (max_idx - min_idx) // 2
                self.eps = eps_range[mid_idx]
                self.labels, self.outlier_proportion_sample = self._dbscan(df, eps=self.eps)
                LOGGER.debug("Attempt on eps = {}: Outliers: {}%"
                             .format(self.eps.round(self.sigfig),
                                     round((100 * self.outlier_proportion_sample), self.sigfig)))
                err = self.target - self.outlier_proportion_sample
                if abs(err) < self.sigma:
                    LOGGER.debug("Final EPS Parameter: {} with Error {}"
                                 .format(self.eps.round(self.sigfig), round(err, self.sigfig)))
                    return self.labels, self.outlier_proportion_sample
                elif err < 0:  # Indicates p < p^, our midpoint has too much pollution
                    return binary_param_search(df, eps_range, mid_idx + 1, max_idx)
                else:  # p > p^, our parameter is too specific
                    return binary_param_search(df, eps_range, min_idx, mid_idx - 1)
            else:
                # noinspection PyBroadException
                try:
                    LOGGER.debug(
                        "No Valid EPS parameters for this run. Returning final runs data with eps=%s & p_hat=%s%%"
                        % (round(self.eps, self.sigfig), round(self.outlier_proportion_sample, self.sigfig)))
                    return self.labels, self.outlier_proportion_sample
                except:
                    LOGGER.debug("Model failed on first iteration, likely due to no valid data")
                    return 1, 0

        return binary_param_search(df, self.eps_range, 0, len(self.eps_range) - 1)

    def _dbscan(self, df, eps):
        labels = DBSCAN(eps=eps).fit_predict(df)
        return labels
