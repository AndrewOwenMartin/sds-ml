import collections, datetime, functools, itertools
import json, logging, pathlib, random, re
from logging import DEBUG, INFO, WARNING, ERROR, FATAL
SILENT = 0

from importlib import reload

import sds_ml.sds_ml as sds_ml
import sds_ml.pima as pima
import sds_ml.variants as variants

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s %(levelname)-4s %(name)s %(message)s",
    style="%",
)

rng = random.Random()
