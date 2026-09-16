import numpy as np
from hashlib import shake_256

from .ring import Rq


def _delta(q, p):
    return q // p


class HomPKE:
    """Normal mode of the FAHE cryptosystem: additively homomorphic Regev
    encryption over R_q = Z_q[X]/(X^n+1)."""

    def __init__(self, ring: Rq, p: int, sigma: float):
        self.R = ring
        self.p = p
        self.sigma = sigma
        self.delta = _delta(ring.q, p)

    def keygen(self, rng, seed=b"simurgh-elec"):
        R = self.R
        a = R.expand(seed, 1)[0]
        s_res, s_raw = R.gaussian(rng, self.sigma)
        e_res, e_raw = R.gaussian(rng, self.sigma)
        b = R.add(R.mul(a, s_res), e_res)
        return (a, b), (s_res, s_raw)

    def encode(self, vec):
        R = self.R
        coeffs = np.zeros(R.n, dtype=np.int64)
        coeffs[: len(vec)] = np.asarray(vec, dtype=np.int64) % self.p
        return coeffs

    def decode(self, poly_int, length):
        q, p = self.R.q, self.p
        out = []
        for i in range(length):
            x = int(poly_int[i]) % q
            out.append(int((x * p + q // 2) // q) % p)
        return out

    def encrypt(self, pk, rng, msg_coeffs):
        R = self.R
        a, b = pk
        r_res, _ = R.gaussian(rng, self.sigma)
        f0_res, _ = R.gaussian(rng, self.sigma)
        f1_res, _ = R.gaussian(rng, self.sigma)
        m_res = R.scale(np.asarray(msg_coeffs, dtype=np.int64), self.delta)
        c0 = R.add(R.mul(a, r_res), f0_res)
        c1 = R.add(R.add(R.mul(b, r_res), f1_res), m_res)
        return np.stack([c0, c1]), (r_res, f0_res, f1_res)

    def encrypt_zero(self, pk, rng, sigma=None):
        return self.encrypt(pk, rng, np.zeros(self.R.n, dtype=np.int64))

    def decrypt(self, sk, ct, length):
        R = self.R
        s_res, _ = sk
        raw = R.sub(ct[1], R.mul(s_res, ct[0]))
        return self.decode(R.to_int(raw), length)

    def noise(self, sk, ct, msg_coeffs):
        R = self.R
        s_res, _ = sk
        raw = R.sub(ct[1], R.mul(s_res, ct[0]))
        m_res = R.scale(np.asarray(msg_coeffs, dtype=np.int64), self.delta)
        return R.infty(R.sub(raw, m_res))

    def add(self, ct1, ct2):
        return self.R.add(ct1, ct2)

    def sub(self, ct1, ct2):
        return self.R.sub(ct1, ct2)

    def mul_plain(self, ct, poly_res):
        return self.R.mul(ct, poly_res[None, ...])


class AnamorphicPKE:
    """FAHE dual-Regev instance: public-key anamorphic (dk is empty) and
    additively homomorphic on both the normal and the covert channel."""

    def __init__(self, ring: Rq, p: int, p_hat: int, sigma: float, sigma_trap: float, m_bar: int):
        self.R = ring
        self.p = p
        self.p_hat = p_hat
        self.sigma = sigma
        self.sigma_trap = sigma_trap
        self.m_bar = m_bar
        self.m = m_bar + 1
        self.delta = _delta(ring.q, p)
        self.delta_hat = _delta(ring.q, p_hat)

    def keygen(self, rng):
        R = self.R
        A = R.uniform(rng, self.m)
        e_res, _ = R.gaussian(rng, self.sigma, self.m)
        u = R.inner(A, e_res)
        return (A, u), e_res, None

    def agen(self, rng, seed=None):
        R = self.R
        seed = seed if seed is not None else rng.integers(0, 1 << 30).tobytes()
        Abar = R.uniform(rng, self.m_bar)
        Rt_res, _ = R.gaussian(rng, self.sigma_trap, self.m_bar)
        last = R.add(R.inner(Abar, Rt_res), R.small_int(np.eye(1, R.n, 0, dtype=np.int64)[0]))
        A = np.concatenate([Abar, last[None, ...]], axis=0)
        e_res, _ = R.gaussian(rng, self.sigma, self.m)
        u = R.inner(A, e_res)
        return (A, u), e_res, Rt_res

    def encode(self, vec, modulus):
        R = self.R
        c = np.zeros(R.n, dtype=np.int64)
        c[: len(vec)] = np.asarray(vec, dtype=np.int64) % modulus
        return c

    def encrypt(self, apk, rng, msg):
        R = self.R
        A, u = apk
        s = R.uniform(rng)
        f0, _ = R.gaussian(rng, self.sigma, self.m)
        f1, _ = R.gaussian(rng, self.sigma)
        c0 = R.add(R.mul(A, s[None, ...]), f0)
        c1 = R.add(R.add(R.mul(u, s), f1), R.scale(np.asarray(msg, dtype=np.int64), self.delta))
        return c0, c1

    def aencrypt(self, apk, rng, msg, covert):
        R = self.R
        A, u = apk
        s_res, _ = R.gaussian(rng, self.sigma)
        shat = R.add(s_res, R.scale(np.asarray(covert, dtype=np.int64), self.delta_hat))
        f0, _ = R.gaussian(rng, self.sigma, self.m)
        f1, _ = R.gaussian(rng, self.sigma)
        c0 = R.add(R.mul(A, shat[None, ...]), f0)
        c1 = R.add(R.add(R.mul(u, shat), f1), R.scale(np.asarray(msg, dtype=np.int64), self.delta))
        return c0, c1

    def decrypt(self, ask, ct, length):
        R = self.R
        c0, c1 = ct
        raw = R.sub(c1, R.inner(ask, c0))
        vals = R.to_int(raw)
        q, p = R.q, self.p
        return [int((int(vals[i]) * p + q // 2) // q) % p for i in range(length)]

    def adecrypt(self, trap, ct, length, robust_bound=None):
        R = self.R
        c0, c1 = ct
        d = R.sub(c0[self.m_bar], R.inner(trap, c0[: self.m_bar]))
        vals = R.center_int(d)
        q, ph = R.q, self.p_hat
        dh = self.delta_hat
        out, resid = [], 0
        for i in range(R.n):
            x = int(vals[i])
            k = int(round(x / dh))
            resid = max(resid, abs(x - k * dh))
            out.append(k % ph)
        if robust_bound is not None and resid > robust_bound:
            return None
        return out[:length]

    def add(self, ct1, ct2):
        return (self.R.add(ct1[0], ct2[0]), self.R.add(ct1[1], ct2[1]))
