import numpy as np
from hashlib import shake_256

from .ring import Rq


def lagrange_at_zero(indices, modulus):
    out = []
    for i in indices:
        num, den = 1, 1
        for j in indices:
            if j == i:
                continue
            num = (num * (-j)) % modulus
            den = (den * (i - j)) % modulus
        out.append(num * pow(den, -1, modulus) % modulus)
    return out


class Submersion:
    """Random submersion: a Johnson-Lindenstrauss projection blinded by a
    Gaussian mask, used as an approximate proof of shortness."""

    def __init__(self, ring: Rq, rows=256, sigma_blind=1 << 16):
        self.R = ring
        self.rows = rows
        self.sigma_blind = sigma_blind

    def project(self, seed, vec_int, rng):
        n = vec_int.size
        raw = shake_256(seed).digest(self.rows * n)
        gamma = (np.frombuffer(raw, dtype=np.uint8).reshape(self.rows, n) % 3).astype(np.int64) - 1
        blind = np.rint(rng.normal(0.0, self.sigma_blind, size=self.rows)).astype(np.int64)
        return gamma @ vec_int.astype(np.int64) + blind, gamma

    def check(self, proj, bound):
        return int(np.max(np.abs(proj))) <= bound


class ThresholdKey:
    """Shamir sharing of the Regev secret key with Renyi-calibrated flooding."""

    def __init__(self, ring: Rq, n_trustees, threshold, sigma_flood):
        self.R = ring
        self.n = n_trustees
        self.t = threshold
        self.sigma_flood = sigma_flood

    def share(self, rng, secret_raw):
        R = self.R
        deg = self.t - 1
        coeffs = [secret_raw]
        for _ in range(deg):
            hi = rng.integers(0, 1 << 32, size=R.n, dtype=np.int64).astype(object)
            lo = rng.integers(0, 1 << 32, size=R.n, dtype=np.int64).astype(object)
            coeffs.append(np.array([(int(h) * (1 << 32) + int(l)) % R.q for h, l in zip(hi, lo)], dtype=object))
        shares = []
        for i in range(1, self.n + 1):
            acc = np.zeros(R.n, dtype=object)
            xp = 1
            for c in coeffs:
                acc = acc + np.asarray(c, dtype=object) * xp
                xp *= i
            shares.append(np.array([int(x) % R.q for x in acc], dtype=object))
        return shares

    def partial_decrypt(self, share_int, c0, rng, index_set, my_index):
        R = self.R
        coeff = lagrange_at_zero(index_set, R.q)[index_set.index(my_index)]
        s_res = R.from_int(np.array([int(x) * coeff % R.q for x in share_int], dtype=object))
        flood = np.rint(rng.normal(0.0, self.sigma_flood, size=R.n)).astype(np.int64)
        return R.add(R.mul(s_res, c0), R.small_int(flood))

    def combine(self, c1, partials):
        R = self.R
        acc = partials[0]
        for p in partials[1:]:
            acc = R.add(acc, p)
        return R.sub(c1, acc)


def renyi_flooding_sigma(noise_bound, n_queries, lam=128):
    """Flooding width from the Renyi-divergence bound of the flood-and-submerse
    framework: sigma >= B * sqrt(2 * lam * Q / log 2)."""
    return float(noise_bound) * float(np.sqrt(2.0 * lam * n_queries / np.log(2.0)))
