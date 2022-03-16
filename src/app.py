import sys
import json
import numpy as np
import pandas as pd
from utils import logger
from utils.constants import *

LOGGER = logger.setup_logger(__name__)


def main(df):
    LOGGER.debug("hello")

    return 1


if __name__ == '__main__':
    df = sys.argv[1]
    main(df)
