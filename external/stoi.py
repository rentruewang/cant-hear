"""
Calculate the STOI score
"""

import common
from pystoi.stoi import stoi as _stoi

stoi = common.evaluate(_stoi)
