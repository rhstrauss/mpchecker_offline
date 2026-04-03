"""Tests for ADES PSV, hldet CSV, auto-detection, and multi-file support."""

import pytest
from mpchecker.obs_parser import (
    parse_ades_psv, parse_hldet, detect_format, parse_auto,
    _parse_ades_obstime,
)


# ---------------------------------------------------------------------------
# ADES PSV tests
# ---------------------------------------------------------------------------

SAMPLE_ADES = """\
# version=2022
# observatory
! mpcCode V00
# submitter
! name J. Smith
permID |provID    |trkSub|mode|stn |obsTime                |ra         |dec        |mag  |band
       |2024 AA   |      |CCD |V00 |2024-01-15T06:00:00.00Z|48.144     |-12.345    |21.5 |r
12345  |          |      |CCD |568 |2024-01-15T12:30:00Z   |180.0      |+45.0      |18.2 |V
       |2024 BB   |      |CCD |T08 |2024-02-01T00:00:00Z   |90.123     |-0.456     |     |
"""


class TestAdesObstime:
    def test_full_iso(self):
        mjd = _parse_ades_obstime('2024-01-15T06:00:00.00Z')
        assert abs(mjd - 60324.25) < 0.001

    def test_no_z(self):
        mjd = _parse_ades_obstime('2024-01-15T12:00:00')
        assert abs(mjd - 60324.5) < 0.001

    def test_no_seconds(self):
        mjd = _parse_ades_obstime('2024-01-15T12:00Z')
        assert abs(mjd - 60324.5) < 0.001


class TestAdesParsing:
    def test_basic_parse(self):
        obs = parse_ades_psv(SAMPLE_ADES)
        assert len(obs) == 3

    def test_provisional_designation(self):
        obs = parse_ades_psv(SAMPLE_ADES)
        assert obs[0].designation == '2024 AA'

    def test_permanent_designation(self):
        obs = parse_ades_psv(SAMPLE_ADES)
        assert obs[1].designation == '12345'

    def test_coordinates(self):
        obs = parse_ades_psv(SAMPLE_ADES)
        assert abs(obs[0].ra_deg - 48.144) < 0.001
        assert abs(obs[0].dec_deg - (-12.345)) < 0.001

    def test_obscode(self):
        obs = parse_ades_psv(SAMPLE_ADES)
        assert obs[0].obscode == 'V00'
        assert obs[1].obscode == '568'
        assert obs[2].obscode == 'T08'

    def test_magnitude(self):
        obs = parse_ades_psv(SAMPLE_ADES)
        assert abs(obs[0].mag - 21.5) < 0.01
        assert obs[0].band == 'r'

    def test_missing_magnitude(self):
        obs = parse_ades_psv(SAMPLE_ADES)
        assert obs[2].mag is None

    def test_mode_as_note2(self):
        obs = parse_ades_psv(SAMPLE_ADES)
        assert obs[0].note2 == 'CCD'

    def test_empty_input(self):
        assert parse_ades_psv('') == []

    def test_header_only(self):
        text = 'permID |provID |stn |obsTime |ra |dec\n'
        assert parse_ades_psv(text) == []


# ---------------------------------------------------------------------------
# hldet CSV tests
# ---------------------------------------------------------------------------

SAMPLE_HLDET = """\
objID,MJD,RA,Dec,mag,band,obscode
2024_AA,60324.25,48.144,-12.345,21.5,r,V00
2024_BB,60324.50,180.0,45.0,18.2,V,568
2024_CC,60355.00,90.123,-0.456,,g,T08
"""

SAMPLE_HLDET_ALT_COLS = """\
Name,FieldMJD,RAdeg,Decdeg,trailedSourceMag,optFilter,obsCode
TestObj,60324.25,48.144,-12.345,21.5,r,V00
"""

SAMPLE_HLDET_JD = """\
det_id,JD,RA,Dec,Vmag,obscode
det001,2460324.75,48.144,-12.345,21.5,V00
"""


class TestHldetParsing:
    def test_basic_parse(self):
        obs = parse_hldet(SAMPLE_HLDET)
        assert len(obs) == 3

    def test_coordinates(self):
        obs = parse_hldet(SAMPLE_HLDET)
        assert abs(obs[0].ra_deg - 48.144) < 0.001
        assert abs(obs[0].dec_deg - (-12.345)) < 0.001
        assert abs(obs[1].ra_deg - 180.0) < 0.001
        assert abs(obs[1].dec_deg - 45.0) < 0.001

    def test_epoch(self):
        obs = parse_hldet(SAMPLE_HLDET)
        assert abs(obs[0].epoch_mjd - 60324.25) < 0.001

    def test_obscode(self):
        obs = parse_hldet(SAMPLE_HLDET)
        assert obs[0].obscode == 'V00'
        assert obs[1].obscode == '568'

    def test_designation(self):
        obs = parse_hldet(SAMPLE_HLDET)
        assert obs[0].designation == '2024_AA'

    def test_magnitude(self):
        obs = parse_hldet(SAMPLE_HLDET)
        assert abs(obs[0].mag - 21.5) < 0.01
        assert obs[2].mag is None

    def test_band(self):
        obs = parse_hldet(SAMPLE_HLDET)
        assert obs[0].band == 'r'
        assert obs[2].band == 'g'

    def test_alt_column_names(self):
        obs = parse_hldet(SAMPLE_HLDET_ALT_COLS)
        assert len(obs) == 1
        assert obs[0].designation == 'TestObj'
        assert abs(obs[0].ra_deg - 48.144) < 0.001

    def test_jd_column(self):
        obs = parse_hldet(SAMPLE_HLDET_JD)
        assert len(obs) == 1
        # JD 2460324.75 = MJD 60324.25
        assert abs(obs[0].epoch_mjd - 60324.25) < 0.001

    def test_empty_input(self):
        assert parse_hldet('') == []

    def test_header_only(self):
        assert parse_hldet('MJD,RA,Dec\n') == []


# ---------------------------------------------------------------------------
# Format auto-detection tests
# ---------------------------------------------------------------------------

class TestFormatDetection:
    def test_detect_ades_version(self):
        assert detect_format('# version=2022\npermID|provID\n') == 'ades'

    def test_detect_ades_pipe_header(self):
        text = 'permID |provID |stn |obsTime |ra |dec\n'
        assert detect_format(text) == 'ades'

    def test_detect_hldet_csv(self):
        assert detect_format('MJD,RA,Dec,mag\n1,2,3,4\n') == 'hldet'

    def test_detect_hldet_alt_columns(self):
        assert detect_format('objID,FieldMJD,RAdeg,Decdeg\n') == 'hldet'

    def test_detect_mpc80_default(self):
        from tests.test_obs_parser import _make_line
        assert detect_format(_make_line()) == 'mpc80'

    def test_detect_empty(self):
        assert detect_format('') == 'mpc80'

    def test_parse_auto_ades(self):
        obs = parse_auto(SAMPLE_ADES)
        assert len(obs) == 3
        assert obs[0].designation == '2024 AA'

    def test_parse_auto_hldet(self):
        obs = parse_auto(SAMPLE_HLDET)
        assert len(obs) == 3
        assert obs[0].designation == '2024_AA'
