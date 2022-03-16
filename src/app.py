import sys
import json
import numpy as np
import pandas as pd
from utils import logger
from utils.constants import *
from model import geohash, transformation, predict

LOGGER = logger.setup_logger(__name__)


def main(df):
    """
    This function takes in an NxM dataframe, subsets by chosen lat/lon columns, fits a 1024 hash map, finds the 99%
    Principle Component count, and then subsets and fits a tuned predictor for the 5% most anomalous geo-locations.

    :param df: NxM Dataframe
    :return: Nx1 Pandas Series with corresponding geo-predictions.
    """

    geo_key = [KEY_LAT, KEY_LON]
    df_latlon = df[geo_key].drop_duplicates(keep='first').reset_index().drop(columns={'index'}).dropna()
    if len(df_latlon) == 0:
        raise Exception("No Latitude/Longitude information in the dataset, all values null.")
    hash_df = geohash.Hash().run(df_latlon)
    hash_df_pca = transformation.pca_cumulative(hash_df)
    df_latlon['geo_cluster_flag'] = predict.SpatialClustering(target=0.05,
                                                              sigma=0.025,
                                                              cv=True).fit_predict(hash_df_pca)

    return df.merge(df_latlon, how='left', left_on=geo_key, right_on=geo_key)['geo_cluster_flag']

