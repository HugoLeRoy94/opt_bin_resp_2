"""Self-consistency of the closed-form homomer MI. No simulation, no data.

Reference: analytics/homomer_array_mi.tex. Equation numbers are that document's.
Nothing downstream is trustworthy until these pass, which is why they are
collected rather than left in a script someone has to remember to run.

Symbols. nu is the single scalar the whole parameter dependence collapses onto;
eta(nu) is the per-receptor information in bits; rho = sigma sqrt(D) / lambda;
Gamma = Emax / sigma_c; w = U^2 / lambda^2. h2 is the binary entropy in bits,
Phibar the standard normal upper tail, W the Lambert function.
"""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from tasks._lib import theory as T


# ── eta(nu): the per-receptor information ───────────────────────────────────

def test_eta_at_one_matches_the_closed_form():
    """Eq. 52: eta(1) = 1 / (2 ln 2) exactly."""
    assert T.eta(1.0) == pytest.approx(T.ETA_AT_ONE, abs=1e-9)


def test_eta_quadrature_is_converged():
    """Refining the quadrature must not move the answer."""
    assert T.eta(1.0) == pytest.approx(T.eta(1.0, 200001), abs=1e-9)


@pytest.mark.parametrize("nu", [0.01, 0.5, 1, 3, 10, 50])
def test_eta_is_a_number_of_bits(nu):
    """One receptor carries between 0 and 1 bit."""
    assert 0.0 <= T.eta(nu) <= 1.0


def test_eta_decreases_with_nu():
    """nu is a noise-to-signal scale, so more of it carries less."""
    grid = [.1, .5, 1, 2, 5, 10]
    assert all(T.eta(a) > T.eta(b) for a, b in zip(grid, grid[1:]))


@pytest.mark.parametrize("nu", [0.05, 0.1, 0.2])
def test_small_nu_asymptote(nu):
    """Eq. 51 holds to 1% below nu = 0.2."""
    assert T.eta(nu) / T.eta_small_nu(nu) == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("nu", [5, 10, 30])
def test_large_nu_asymptote(nu):
    """Eq. 54 holds to 5% above nu = 5."""
    assert T.eta(nu) / T.eta_large_nu(nu) == pytest.approx(1.0, abs=0.05)


def test_constant_A_matches_the_document():
    """The tex quotes A = 2.606 from a coarse Simpson rule. Recompute it.

    T._h2 defines h2(0) = h2(1) = 0; an unguarded 0*log2(0) returns NaN here.
    """
    A = quad(lambda x: float(T._h2(np.array([norm.sf(x)]))[0]), -40, 40, limit=800)[0]
    assert A == pytest.approx(2.6060764896686517, abs=1e-6)


# ── MI: the count distribution ──────────────────────────────────────────────

@pytest.mark.parametrize("nu", [0.01, 0.5, 1.0, 3.0, 10.0])
@pytest.mark.parametrize("R", [2, 6, 12, 20, 40])
def test_mi_is_normalised_bounded_and_consistent(nu, R):
    """Eq. 48: MI = R eta(nu) + I(A;q), and R binary outputs carry at most R bits."""
    d = T.mi(nu, R)
    assert d["norm_err"] < 1e-10
    assert d["MI"] <= R + 1e-9
    assert d["MI"] == pytest.approx(d["R_eta"] + d["I_A_q"], abs=1e-9)


def test_mi_saturates_at_R_when_noise_vanishes():
    """nu -> 0 means every receptor is a clean independent bit."""
    assert T.mi(1e-3, 12)["MI"] == pytest.approx(12, abs=1e-3)


def test_mi_collapses_to_about_one_bit_when_noise_dominates():
    """nu -> inf leaves only the shared count, worth about a bit."""
    assert T.mi(1e4, 12)["MI"] == pytest.approx(1.0, abs=0.15)


# ── the delta route against the exact route ─────────────────────────────────

@pytest.mark.parametrize("D,sigma,lam,Emax,sigma_c", [
    (10, 1, 5, 10, 1), (10, 1, 5, 100, 1), (8, 1, 4, 10, 1), (10, 1, 2, 10, 1)])
def test_lambert_and_exact_routes_agree_where_predicted(D, sigma, lam, Emax, sigma_c):
    """Eq. 41: the delta-method expansion of ln Z is good when selfcons >> 1.

    The exact route drops that expansion and keeps the saturation atoms, so
    where selfcons is small they are ALLOWED to disagree and the check is
    skipped rather than relaxed.
    """
    p = T.Params(D=D, sigma=sigma, lam=lam, Emax=Emax, sigma_c=sigma_c)
    lambert, exact = T.lambert_optimum(p), T.exact_optimum(p)
    if lambert["selfcons"] <= 3:
        pytest.skip(f"selfcons = {lambert['selfcons']:.2f}; the expansion is not "
                    f"claimed to hold here")
    assert exact["U_star"] / lambert["U_star"] == pytest.approx(1.0, abs=0.05)
    assert exact["eta"] / T.eta(lambert["nu_min"]) == pytest.approx(1.0, abs=0.02)


def test_the_exact_route_balances_firing_at_one_half():
    """Ebase is chosen so the population fires half the time."""
    p = T.Params(D=8, sigma=1, lam=4, Emax=10, sigma_c=1)
    assert T.exact_optimum(p)["mean_firing"] == pytest.approx(0.5, abs=1e-6)


# ── why the kernel has to saturate ──────────────────────────────────────────

def test_the_quadratic_kernel_has_no_optimum():
    """Eq. 34: with a quadratic kernel nu falls monotonically, so there is no
    best width and the model is degenerate. This is why the kernel saturates."""
    p = T.Params(D=8, sigma=1, lam=4, Emax=10, sigma_c=1)
    w = np.array([0.1, 0.5, 1, 2, 4, 8, 16])
    assert np.all(np.diff(T.nu_quadratic_of_w(p, w)) < 0)


def test_the_gaussian_kernel_has_an_interior_optimum():
    """The saturating kernel gives nu a genuine minimum in w, which is the
    operating point the array is predicted to find."""
    p = T.Params(D=8, sigma=1, lam=4, Emax=10, sigma_c=1)
    w = np.array([0.1, 0.5, 1, 2, 4, 8, 16])
    assert int(np.argmin(T.nu_of_w(p, w))) not in (0, len(w) - 1)
