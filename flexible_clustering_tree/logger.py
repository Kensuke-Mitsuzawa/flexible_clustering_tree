# -*- coding: utf-8 -*-

from logging import getLogger, StreamHandler, FileHandler, Logger
import logging
import sys

custmoFormatter = logging.Formatter(
    fmt='[%(asctime)s]%(levelname)s - %(filename)s#%(funcName)s:%(lineno)d: %(message)s',
    datefmt='Y/%m/%d %H:%M:%S'
)

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(custmoFormatter)

logger_name = 'flexible-clustering-tree'
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False
