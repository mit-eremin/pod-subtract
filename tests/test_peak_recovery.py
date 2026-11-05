import numpy as np
from bgsub.validation import peaks as PK
from bgsub.vendor import binning as B


def test_spot_sums_to_total():
    y0, x0, lam = PK.spot_pixels(100.3, 200.7, 0.8, 1000.0, 10)
    assert abs(lam.sum() - 1000.0) < 1e-6
    r = PK.patch_radius(y0, x0, lam.shape[0], 100.3, 200.7)
    box, ann = PK.box_and_annulus(r, 3.0, 5.0, 9.0)
    assert 0.99 < lam[box].sum() / 1000 <= 1.0 and not (box & ann).any()


def test_quantiser_prob_normalised():
    for tie in (0.35, None):
        for N in range(0, 30):
            p = sum(PK.quantiser_prob(np.array(N), np.array(K), 4, tie) for K in range(0, 10))
            assert abs(p - 1) < 1e-12


def test_quantiser_matches_quantise():
    rng = np.random.default_rng(1)
    N = rng.integers(0, 40, 200000); K = B.quantise(N, 8, 0.35, rng=rng)
    for n in (2, 6, 7, 10):
        k = K[N == n]
        for kk in np.unique(k):
            assert abs((k == kk).mean() - PK.quantiser_prob(np.array(n), np.array(kk), 4, 0.35)) < 0.02


def test_latent_posterior_recovers_poisson_marginal():
    rng = np.random.default_rng(2)
    for mu in (0.3, 1.5, 6.0):
        N = rng.poisson(mu, 200000); K = B.quantise(N, 8, 0.35, rng=rng)
        Ns = PK.sample_latent(K, np.full(K.shape, mu), 8, 0.35, rng)
        assert np.all(PK.quantiser_prob(Ns, K, 4, 0.35) > 0)
        assert abs(Ns.mean() / mu - 1) < 0.01 and abs(Ns.var() / mu - 1) < 0.03


def test_inject_zero_is_identity_and_bin4_adds():
    rng = np.random.default_rng(3)
    K = B.quantise(rng.poisson(1.0, (21, 21)), 8, 0.35, rng=rng)
    assert np.array_equal(PK.inject(K, np.ones((21, 21)), np.zeros((21, 21)), 8, 0.35, 0.0, rng), K)
    K4 = rng.poisson(1.0, (21, 21))
    out = PK.inject(K4, np.ones((21, 21)), np.full((21, 21), 1e-12), 4, None, 0.0, rng)
    assert np.array_equal(out, K4)


def test_inject_mean_linear_at_high_counts():
    rng = np.random.default_rng(4)
    mu = 20.0; lam = 40.0; n = 50000
    K = B.quantise(rng.poisson(mu, n), 8, 0.35, rng=rng)
    Kp = PK.inject(K, np.full(n, mu), np.full(n, lam), 8, 0.35, 0.0, rng)
    assert abs(4 * (Kp - K).mean() / lam - 1) < 0.01


def test_integrate_annulus_flat():
    g = PK.StoredVariance(4, None)
    r = PK.patch_radius(0, 0, 21, 10, 10); box, ann = PK.box_and_annulus(r, 3.0, 5.0, 9.0)
    Kp = np.full((21, 21), 2); Kp[box] += 5
    I, sn, sl, Bb = PK.integrate(Kp, None, box, ann, 1.0, g)
    assert I == 5 * box.sum() and Bb == 2 * box.sum() and sl == sn


def test_pb_star():
    ladder = [0.1, 1, 10]
    cells = {0.1: ([1.3, 1.2], 1.25), 1: ([1.01, 0.99, 1.2], 1.0), 10: ([1.0, 1.0], 1.0)}
    assert PK.pb_star(cells, ladder, 0.05) == 1
    cells[1] = ([1.2, 1.2, 1.0], 1.0)
    assert PK.pb_star(cells, ladder, 0.05) == 10


def test_profile_at_convention():
    P = np.arange(10, dtype=float)
    assert PK.profile_at(P, np.array([3.5]))[0] == 3.0
