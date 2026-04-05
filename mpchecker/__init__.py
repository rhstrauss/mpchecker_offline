"""
mpchecker — local replication of MPC MPChecker with pyoorb orbit propagation
and planetary satellite support via SPICE kernels.
"""

__version__ = '0.1.0'

from .obs_parser import (parse_observations, parse_file, Observation,
                         parse_ades_psv, parse_ades_file,
                         parse_hldet, parse_hldet_file,
                         parse_auto, parse_file_auto, detect_format)
from .checker import (check_observations, identify_tracklet,
                      CheckResult, Match, Identification)
