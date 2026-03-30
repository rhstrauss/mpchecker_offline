"""Tests for MPC 80-column observation parser."""

import pytest
from mpchecker.obs_parser import (
    parse_line, parse_observations, parse_date_mjd,
    parse_ra_deg, parse_dec_deg, unpack_number, unpack_provisional,
)


def _make_line(num='00001', prov='       ', disc=' ', n1=' ', n2='C',
               date='2024 01 15.12345 ', ra='03 12 34.560',
               dec='+12 34 56.70', blank1='         ',
               mag=' 10.5', band='V', blank2='      ', obs='695'):
    line = num + prov + disc + n1 + n2 + date + ra + dec + blank1 + mag + band + blank2 + obs
    assert len(line) == 80, f'Bad length: {len(line)}'
    return line


class TestDateParsing:
    def test_standard_date(self):
        mjd = parse_date_mjd('2024 01 15.12345 ')
        assert abs(mjd - 60324.12345) < 1e-4

    def test_fractional_day(self):
        mjd = parse_date_mjd('2000 01 01.50000 ')
        # 2000 Jan 1 noon = JD 2451545.0 → MJD 51544.5
        assert abs(mjd - 51544.5) < 1e-4

    def test_day_zero_fraction(self):
        mjd = parse_date_mjd('2000 01 01.00000 ')
        assert abs(mjd - 51544.0) < 1e-4


class TestCoordinateParsing:
    def test_ra_noon(self):
        # 12h 00m 00.000s = 180.0 deg
        ra = parse_ra_deg('12 00 00.000')
        assert abs(ra - 180.0) < 1e-6

    def test_ra_origin(self):
        ra = parse_ra_deg('00 00 00.000')
        assert abs(ra) < 1e-6

    def test_ra_max(self):
        # 23h 59m 59.999s ≈ 359.999 deg
        ra = parse_ra_deg('23 59 59.999')
        assert 359.99 < ra < 360.0

    def test_dec_positive(self):
        dec = parse_dec_deg('+45 30 00.00')
        assert abs(dec - 45.5) < 1e-6

    def test_dec_negative(self):
        dec = parse_dec_deg('-90 00 00.00')
        assert abs(dec + 90.0) < 1e-6

    def test_dec_zero(self):
        dec = parse_dec_deg('+00 00 00.00')
        assert abs(dec) < 1e-6


class TestParseLine:
    def test_basic_asteroid(self):
        line = _make_line()
        obs = parse_line(line)
        assert obs is not None
        assert obs.designation == '1'
        assert obs.obscode == '695'
        assert abs(obs.ra_deg - 48.144) < 0.001
        assert abs(obs.dec_deg - 12.582) < 0.001
        assert abs(obs.mag - 10.5) < 0.01
        assert obs.band == 'V'
        assert obs.obj_type == 'minor_planet'
        assert obs.note2 == 'C'

    def test_provisional_designation(self):
        line = _make_line(num='     ', prov='K07Tf8A')
        obs = parse_line(line)
        assert obs is not None
        assert obs.packed_desig == 'K07Tf8A'

    def test_discovery_asterisk(self):
        line = _make_line(disc='*')
        obs = parse_line(line)
        assert obs.discovery is True

    def test_no_magnitude(self):
        line = _make_line(mag='     ', band=' ')
        obs = parse_line(line)
        assert obs.mag is None
        assert obs.band is None

    def test_negative_dec(self):
        line = _make_line(dec='-45 30 00.00')
        obs = parse_line(line)
        assert obs.dec_deg < 0

    def test_blank_line_returns_none(self):
        assert parse_line('') is None
        assert parse_line('   ' * 27) is None

    def test_comment_returns_none(self):
        assert parse_line('# comment') is None
        assert parse_line('! comment') is None


class TestDesignationUnpacking:
    def test_numbered_simple(self):
        assert unpack_number('00001') == '1'
        assert unpack_number('00433') == '433'
        assert unpack_number('99999') == '99999'

    def test_numbered_large(self):
        # A0000 = 10*10000 = 100000
        assert unpack_number('A0000') == '100000'

    def test_provisional_basic(self):
        # J95X00A = 1995 XA
        result = unpack_provisional('J95X00A')
        assert '1995' in result
        assert 'X' in result


class TestParseObservations:
    def test_multiple_lines(self):
        lines = [
            _make_line(num='00001', obs='568'),
            _make_line(num='00433', obs='695'),
            '',  # blank line should be skipped
            '# comment should be skipped',
        ]
        obs_list = parse_observations('\n'.join(lines))
        assert len(obs_list) == 2
        assert obs_list[0].designation == '1'
        assert obs_list[1].designation == '433'

    def test_obscode_from_line(self):
        line = _make_line(obs='T08')
        obs_list = parse_observations(line)
        assert obs_list[0].obscode == 'T08'
