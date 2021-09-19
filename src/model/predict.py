from sklearn.cluster import DBSCAN
from src.model import geohash, transformation
from src.utils import logger

LOGGER = logger.setup_logger(__name__)


def hash_predict(df, **kwargs):
    """
    This function takes in an NxM dataframe, subsets by chosen lat/lon columns, fits a 1024 hash map, finds the 99%
    Principle Component count, and then subsets and fits a tuned predictor for the 5% most anomalous geo-locations.

    This function is motivated due to the fact this same process takes place in every single hunt. We will always want
    to subset to that datasets lat/lon, establish a DB connection, generate/import a hash, then fit the same model.

    :param df: NxM Dataframe
    :param kwargs: Contains the following keys
        user_key: How the overall model is keyed (e.g.: ['user', 'src_ip']), must be type List
        geo_key: Latitude/Longtidue fields. Expects a len 2 list (e.g.: ['dst_geo.latitude', 'dst_geo.longitude'])
        connection: SQL DB Connection
        source_index: SQL DB Index
        local_test: Boolean where if True, it forgoes a DB connection and uses a local docker pickle path
    :return: Nx1 Pandas Series with corresponding geo-predictions.
    """
    geo_key = kwargs.get('geo_key')
    df_latlon = df[geo_key].drop_duplicates(keep='first').reset_index().drop(columns={'index'}).dropna()
    if len(df_latlon) == 0:
        raise Exception("No Latitude/Longitude information in the dataset, all values null.")
    hash_df = geohash.Hash(key=geo_key,
                           connection=kwargs.get('connection', None),
                           source_index=kwargs.get('source_index', ""),
                           local_test=kwargs.get('local_test', False)
                           ).run(df_latlon)

    hash_df_pca = transformation.pca_cumulative(hash_df)
    df_latlon['geo_cluster_flag'] = SpatialClustering(target=0.05,
                                                      sigma=0.025,
                                                      cv=True).fit_predict(hash_df_pca)

    return df.merge(df_latlon, how='left', left_on=geo_key, right_on=geo_key)['geo_cluster_flag']


class SpatialClustering(object):
    def __init__(self, target, sigma, cv):
        self.target = target
        self.sigma = sigma
        self.cv = cv
        self.sigfig = 4
        self.eps_range = []  # TODO: Need a way to fill this out with reasonable values for eps... how did i do that...?

    def fit_predict(self, df):
        labels, proportion = self._auto_optimize_model(df)
        # TODO: log_proportions_here
        return labels

    ## TODO: THREAD UP ALL THIS WEIRD SELF.SHIT, JESUS FUCK WHAT WERE YOU THINKING. SE A CONSTANTS FILE YOU FUCKING MANIAC.U
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
