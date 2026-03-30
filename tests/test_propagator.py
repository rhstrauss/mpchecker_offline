"""Tests for orbit propagation (Keplerian pre-filter and pyoorb precise)."""

import numpy as np
import pytest
from mpchecker.propagator import (
    solve_kepler, kep_to_radec, ang_sep_deg, get_earth_helio,
    build_oorb_orbits_kep, oorb_ephemeris, _DEG2RAD,
)


class TestKeplerSolver:
    def test_zero_eccentricity(self):
        M = np.array([1.0, 2.0, 3.0])
        e = np.zeros(3)
        E = solve_kepler(M, e)
        np.testing.assert_allclose(E, M, atol=1e-10)

    def test_low_eccentricity(self):
        M = np.array([0.5])
        e = np.array([0.1])
        E = solve_kepler(M, e)
        # Verify M = E - e*sin(E)
        residual = M - (E - e * np.sin(E))
        assert np.max(np.abs(residual)) < 1e-10

    def test_high_eccentricity(self):
        # Comet-like orbit: e=0.95
        M = np.array([0.2])
        e = np.array([0.95])
        E = solve_kepler(M, e)
        residual = M - (E - e * np.sin(E))
        assert np.max(np.abs(residual)) < 1e-8

    def test_batch_convergence(self):
        n = 1000
        M = np.random.uniform(0, 2*np.pi, n)
        e = np.random.uniform(0, 0.98, n)
        E = solve_kepler(M, e)
        residual = M - (E - e * np.sin(E))
        assert np.max(np.abs(residual)) < 1e-8


class TestAngularSeparation:
    def test_zero_separation(self):
        sep = ang_sep_deg(np.array([45.0]), np.array([20.0]), 45.0, 20.0)
        assert sep[0] < 1e-10

    def test_90_degree_separation(self):
        sep = ang_sep_deg(np.array([0.0]), np.array([0.0]), 90.0, 0.0)
        assert abs(sep[0] - 90.0) < 1e-6

    def test_pole_to_equator(self):
        sep = ang_sep_deg(np.array([0.0]), np.array([90.0]), 180.0, 0.0)
        assert abs(sep[0] - 90.0) < 1e-6


class TestEarthPosition:
    def test_reasonable_distance(self):
        """Earth should be ~1 AU from Sun."""
        pos = get_earth_helio(51544.5)  # J2000.0
        dist = np.linalg.norm(pos)
        assert 0.98 < dist < 1.02

    def test_reproducible(self):
        p1 = get_earth_helio(60000.0)
        p2 = get_earth_helio(60000.0)
        np.testing.assert_array_equal(p1, p2)


class TestKeplerianPropagation:
    """Test the fast Keplerian RA/Dec predictor against pyoorb."""

    # Ceres elements at MJD 60222 (2023-Oct-05) TT
    _a     = np.array([2.76786])
    _e     = np.array([0.07934])
    _i     = np.array([10.594 * _DEG2RAD])
    _Omega = np.array([80.305 * _DEG2RAD])
    _omega = np.array([73.597 * _DEG2RAD])
    _M     = np.array([91.893 * _DEG2RAD])
    _epoch = np.array([60222.0])

    def test_ceres_position_in_ballpark(self):
        """Keplerian Ceres position should be within ~1 deg of pyoorb result."""
        t = 60222.0
        obs_helio = get_earth_helio(t)
        ra, dec, dist = kep_to_radec(
            self._a, self._e, self._i, self._Omega,
            self._omega, self._M, self._epoch, t, obs_helio)
        # pyoorb gives RA=237.39, Dec=-19.20 at this epoch
        assert abs(ra[0] - 237.39) < 2.0
        assert abs(dec[0] + 19.20) < 2.0
        assert 2.5 < dist[0] < 4.5

    def test_propagation_step_consistency(self):
        """Propagating 0 days should give same result as epoch position."""
        t = 60222.0
        obs_helio = get_earth_helio(t)
        ra0, dec0, _ = kep_to_radec(
            self._a, self._e, self._i, self._Omega,
            self._omega, self._M, self._epoch, t, obs_helio)
        ra1, dec1, _ = kep_to_radec(
            self._a, self._e, self._i, self._Omega,
            self._omega, self._M, self._epoch, t + 0.0, obs_helio)
        assert abs(ra0[0] - ra1[0]) < 1e-8
        assert abs(dec0[0] - dec1[0]) < 1e-8


class TestPyoorbEphemeris:
    """Test pyoorb precise ephemeris."""

    def test_ceres_ephemeris(self):
        """Ceres V-mag at ~3.4 AU should be around 7-10."""
        orbits = build_oorb_orbits_kep(
            a=np.array([2.76786]),
            e=np.array([0.07934]),
            i_rad=np.array([10.594 * _DEG2RAD]),
            Omega_rad=np.array([80.305 * _DEG2RAD]),
            omega_rad=np.array([73.597 * _DEG2RAD]),
            M_rad=np.array([91.893 * _DEG2RAD]),
            epoch_mjd=np.array([60222.0]),
            H=np.array([3.53]),
            G=np.array([0.12]),
        )
        eph = oorb_ephemeris(orbits, 60222.0, '568')
        ra, dec = float(eph[0, 1]), float(eph[0, 2])
        vmag = float(eph[0, 9])
        r_helio = float(eph[0, 7])
        delta = float(eph[0, 8])

        assert 230 < ra < 245
        assert -25 < dec < -10
        assert 5.0 < vmag < 12.0
        assert 2.5 < r_helio < 3.5
        assert 2.0 < delta < 4.5

    def test_ephemeris_shape(self):
        """Output should have shape [n_orbits, 11]."""
        n = 3
        orbits = build_oorb_orbits_kep(
            a=np.full(n, 2.77),
            e=np.full(n, 0.079),
            i_rad=np.full(n, 10.0 * _DEG2RAD),
            Omega_rad=np.full(n, 80.0 * _DEG2RAD),
            omega_rad=np.full(n, 73.0 * _DEG2RAD),
            M_rad=np.full(n, 90.0 * _DEG2RAD),
            epoch_mjd=np.full(n, 60222.0),
            H=np.full(n, 3.5),
            G=np.full(n, 0.15),
        )
        eph = oorb_ephemeris(orbits, 60222.0, '568')
        assert eph.shape == (n, 11)
