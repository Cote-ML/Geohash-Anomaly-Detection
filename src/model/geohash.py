import pandas as pd
import numpy as np
import math
import sys
import json
from functools import lru_cache
from geopy import distance
import pygeohash
from src.utils import logger
from src.utils.decorator import timing
from src.utils.constants import KEY_LAT, KEY_LON

LOGGER = logger.setup_logger(__name__)


class Hash(object):
    def __init__(self):
        self.tau = 2 * math.pi
        self.sigma = 0.3989422804014327
        self.hash_alphabet = "0123456789bcdefghjkmnpqrstuvwxyz"
        self.earth_circumference_meters = 40.07e6
        self.earth_circumference_half = self.earth_circumference_meters / 2

    def run(self, df):
        hash_table_dict = self._geospatial_hash(df)
        LOGGER.debug(
            "Hash Table Dictionary Created. Converting to 1024 hash pandas dataframe with {0} observations".format(
                len(hash_table_dict)))
        hash_table_df = pd.DataFrame.from_records(hash_table_dict)
        LOGGER.debug("Geo DataFrame Successfully Built . . .")
        return hash_table_df

    def _geospatial_hash(self, df):
        geo_fields = [KEY_LAT, KEY_LON]
        geo_lookup = self._get_all_map()
        geo_df = df[geo_fields].fillna(
            {KEY_LAT: 38.9072, KEY_LON: 77.0369})  # Washington DC as default "safe" value for internal IPs
        hash_map = geo_df.apply(lambda x: self._calc_hash_distances(x[KEY_LAT], x[KEY_LON], geo_lookup), axis=1)
        return hash_map

    def _get_normal_value(self, val, sigma=None, mu=0):
        if sigma is None:
            sigma = self.sigma
        rotation = 1 if val < 0 else -1
        return (1.0 / (math.sqrt(self.tau * (sigma ** 2)))) * (math.e ** (rotation * ((val - mu) / (2 * sigma))))

    def _two_letter_hash(self, lat, lon):
        pyhash = pygeohash.encode(lat, lon, precision=2)
        return pyhash

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
        # Todo: Import geohash here if one exists for a data population. e.g., 
        # geo_hash = self._pull_db_hash()
        geo_hash = self._gen_hash_features()
        # Todo: Save this into a local DB for repeated function calls. e.g., 
        # self._save_geo_hash(geo_hash)
        return geo_hash

    @timing
    def _gen_hash_features(self):
        self.geo_hash_to_lat_lon = self._generate_geo_hash()
        output = {}
        for feature, latLon in self.geo_hash_to_lat_lon.items():
            output[feature] = self._calc_hash_distances(lat=latLon[0], lon=latLon[1], feature_map={})
        return output

    def _generate_geo_hash(self):
        geo_hash_to_lat_long = {}
        for letter1 in self.hash_alphabet:
            for letter2 in self.hash_alphabet:
                code = letter1 + letter2
                lat_long_position = pygeohash.decode(code)
                geo_hash_to_lat_long[code] = lat_long_position
        return geo_hash_to_lat_long

    def _save_geo_hash(self, output):
        LOGGER.debug("Attempting to save Geo Hash into DB")
        try:
            cursor = self.connection.cursor()
            postgres_sql_save_query = "INSERT INTO geo_mappings(index, geo_mapping) VALUES(\'{0}\', \'{1}\' )". \
                format(self.source_index, json.dumps(output))
            cursor.execute(postgres_sql_save_query)
            self.connection.commit()
            LOGGER.debug("Successfully saved Geo Mapping Data into psql")
            return None

        except Exception as error:
            LOGGER.info("failed to save geoMapping data to psql. ")
            LOGGER.error(error)
            return None
            
    def _pull_db_hash(self):
        cursor = self.connection.cursor()
        postgres_sql_select_query = "SELECT geo_mapping FROM geo_mappings WHERE index = \'{0}\'".format(
            self.source_index)
        cursor.execute(postgres_sql_select_query)
        geo_json_rows = cursor.fetchall()
        if len(geo_json_rows) > 0:
            return geo_json_rows[0][0]
        else:
            raise Exception("Database Connection Established, but no data in table.")

