"""
mpchecker — local replication of MPC MPChecker with pyoorb orbit propagation
and planetary satellite support via SPICE kernels.
"""

__version__ = '0.1.0'

from .obs_parser import parse_observations, parse_file, Observation
from .checker import check_observations, CheckResult, Match
