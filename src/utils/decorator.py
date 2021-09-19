from time import time
from functools import wraps
from src.utils import logger


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        log = logger.setup_logger(f.__qualname__)
        result = f(*args, **kw)
        te = time()
        log.debug('runtime: %2.4f sec' % (te-ts))
        return result
    return wrap
