# -*- coding: utf-8 -*-

from pathlib import Path
import logging

__version__ = "0.1.0"

log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parents[0]
DIRS = {
    "ROOT": ROOT_DIR,
    "DATA": ROOT_DIR / "data",
}
