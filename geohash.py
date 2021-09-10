import pandas as pd
import numpy as np
import math


def _auto_optimize_model(self, df):
    def binary_param_search(df, eps_range, min_idx, max_idx):
        if max_idx > min_idx:
            mid_idx = min_idx + (max_idx - min_idx) // 2
            self.eps = eps_range[mid_idx]
            self.labels, self.outlier_proportion_sample = self._dbscan(df, eps=self.eps)
            log.debug("Attempt on eps = {}: Outliers: {}%"
                      .format(self.eps.round(self.sigfig), round((100 * self.outlier_proportion_sample), self.sigfig)))
            err = self.target - self.outlier_proportion_sample
            if abs(err) < self.sigma:
                log.debug("Final EPS Parameter: {} with Error {}"
                          .format(self.eps.round(self.sigfig), round(err, self.sigfig)))
                return self.labels, self.outlier_proportion_sample
            elif err < 0:  # Indicates p < p^, our midpoint has too much pollution
                return binary_param_search(df, eps_range, mid_idx + 1, max_idx)
            else:  # p > p^, our parameter is too specific
                return binary_param_search(df, eps_range, min_idx, mid_idx - 1)
        else:
            # noinspection PyBroadException
            try:
                log.debug("No Valid EPS parameters for this run. Returning final runs data with eps=%s & p_hat=%s%%"
                          % (round(self.eps, self.sigfig), round(self.outlier_proportion_sample, self.sigfig)))
                return self.labels, self.outlier_proportion_sample
            except:
                log.debug("Model failed on first iteration, likely due to no valid data")
                return 1, 0

    return binary_param_search(df, self.eps_range, 0, len(self.eps_range) - 1)


def hash_predict(df, kwargs):
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
        raise exceptions.MissingDataError("No Latitude/Longitude information in the dataset, all values null.")
    hash_df = geohash.Hash(key=geo_key,
                           connection=kwargs.get('connection', None),
                           source_index=kwargs.get('source_index', ""),
                           local_test=kwargs.get('local_test', False)
                           ).run(df_latlon)

    hash_df_pca = transformation.pca_cumulative(hash_df)
    df_latlon['geo_cluster_flag'] = unsupervised.SpatialClustering(target=0.05,
                                                                   sigma=0.025,
                                                                   cv=True).fit_predict(hash_df_pca)
    return df.merge(df_latlon, how='left', left_on=geo_key, right_on=geo_key)['geo_cluster_flag']


class Hash(object):
    def __init__(self, key, connection, source_index, local_test=False):
        self.key = key
        self.connection = connection
        self.source_index = source_index
        self.local_test = local_test
        self.tau = 2 * math.pi
        self.sigma = 0.3989422804014327
        self.hash_alphabet = "0123456789" if local_test else "0123456789bcdefghjkmnpqrstuvwxyz"
        self.earth_circumference_meters = 40.07e6
        self.earth_circumference_half = self.earth_circumference_meters / 2

    def run(self, df):
        hash_table_dict = self._geospatial_hash(df)
        log.debug(
            "Hash Table Dictionary Created. Converting to 1024 hash pandas dataframe with {0} observations".format(
                len(hash_table_dict)))
        hash_table_df = pd.DataFrame.from_records(hash_table_dict)
        log.debug("Geo DataFrame Successfully Built . . .")
        return hash_table_df

    def _geospatial_hash(self, df):
        lat = self.key[0]
        lon = self.key[1]
        geo_fields = [lat, lon]
        for c in geo_fields:
            if c not in df.columns:
                log.debug("No Latitude/Longitude Information. Breaking.")
                return df

        geo_lookup = self._get_all_map()
        geo_df = df[geo_fields].fillna(
            {lat: 38.9072, lon: 77.0369})  # Washington DC as default "safe" value for internal IPs
        hash_map = geo_df.apply(lambda x: self._calc_hash_distances(x[lat], x[lon], geo_lookup), axis=1)
        return hash_map

    def _get_normal_value(self, val, sigma=None, mu=0):
        if sigma is None:
            sigma = self.sigma
        rotation = 1 if val < 0 else -1
        return (1.0 / (math.sqrt(self.tau * (sigma ** 2)))) * (math.e ** (rotation * ((val - mu) / (2 * sigma))))

    def _two_letter_hash(self, lat, lon):
        hash = geohash.encode(lat, lon, precision=2)
        return hash

    def _calc_hash_distances(self, lat, lon, feature_map, prefix=""):
        feature = self._two_letter_hash(lat, lon)
        output = {}
        if feature in feature_map:
            for dst_feature, encoding in feature_map[feature].items():
                output[prefix + dst_feature] = encoding
            return output
        output = self._calc_distance_for_feature(feature, lat, lon, prefix)
        return output

    @lru_cache(maxsize=1024)
    def _calc_distance_for_feature(self, feature, lat, lon, prefix=""):
        output = {}
        for dst_feature, dst_lat_long in self.geo_hash_to_lat_lon.items():
            if dst_feature == feature:
                dist = 0.0
            else:
                try:
                    dist = distance.distance((lat, lon), dst_lat_long).meters
                except ValueError as ve:
                    sys.stderr.write(
                        "Failed to calculate distance for [%s][%s] to [%s][%s] due to [%s] falling back to great circle distance\n" % (
                            feature, (lat, lon), dst_feature, dst_lat_long, ve))
                    dist = distance.great_circle((lat, lon), dst_lat_long).meters

            scaled_distribution = lambda val: self._get_normal_value(val)
            encoding = scaled_distribution(dist / self.earth_circumference_half)
            output[prefix + dst_feature] = encoding
        return output

    def _get_all_map(self):
        try:
            geo_hash = self._pull_db_hash()
        except:
            log.info("source_index has no geo_hash data saved, about to create and attempt to save geo data")
            geo_hash = self._gen_hash_features()
            self._save_geo_hash(geo_hash)
        return geo_hash

    def _pull_db_hash(self):
        cursor = self.connection.cursor()
        postgres_sql_select_query = "SELECT geo_mapping FROM geo_mappings WHERE hunt_index = \'{0}\'".format(
            self.source_index)
        cursor.execute(postgres_sql_select_query)
        geo_json_rows = cursor.fetchall()
        if len(geo_json_rows) > 0:
            return geo_json_rows[0][0]
        else:
            raise exceptions.MissingDataError("Database Connection Established, but no data in table.")

    @timing
    def _gen_hash_features(self):
        self.geo_hash_to_lat_lon = self._generate_geo_hash()
        output = {}
        for feature, latLon in tqdm(self.geo_hash_to_lat_lon.items()):
            output[feature] = self._calc_hash_distances(lat=latLon[0], lon=latLon[1], feature_map={})
        return output

    def _generate_geo_hash(self):
        geo_hash_to_lat_long = {}
        for letter1 in self.hash_alphabet:
            for letter2 in self.hash_alphabet:
                code = letter1 + letter2
                lat_long_position = geohash.decode(code)
                geo_hash_to_lat_long[code] = lat_long_position
        return geo_hash_to_lat_long

    def _save_geo_hash(self, output):
        log.debug("Attempting to save Geo Hash into DB")
        try:
            cursor = self.connection.cursor()
            postgres_sql_save_query = "INSERT INTO geo_mappings(hunt_index, geo_mapping) VALUES(\'{0}\', \'{1}\' )". \
                format(self.source_index, json.dumps(output))
            cursor.execute(postgres_sql_save_query)
            self.connection.commit()
            log.debug("Successfully saved Geo Mapping Data into psql")
            return None

        except Exception as error:
            log.info("failed to save geoMapping data to psql. "
                     "Chances are another worker saved geo data for this hunt, before this worker was able to")
            log.error(error)
            return None


def pca_cumulative(df, epsilon=1):
    cum_var = np.cumsum(
        np.round(decomposition.PCA(n_components=min(df.shape)).fit(df).explained_variance_ratio_, decimals=4) * 100)
    cum_var_diff = np.array([round(j - i, 2) for i, j in zip(cum_var[:-1], cum_var[1:])])
    cutoff = np.argmax(cum_var_diff < epsilon) + 1
    log.debug("Number Components for Convergence of PCA Variance: {}".format(cutoff))
    pca_df = pd.DataFrame(decomposition.PCA(n_components=cutoff).fit_transform(df))
    return pca_df


