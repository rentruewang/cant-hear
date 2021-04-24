"""
Calculate the PESQ score
"""

import common
from pypesq import pesq as _pesq

pesq = common.evaluate(_pesq)
