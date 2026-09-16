import numpy as np
from hashlib import shake_256

from .ring import Rq

TAU = 60


def sample_challenge(ring: Rq, seed: bytes):
    n = ring.n
    raw = shake_256(seed).digest(16 * TAU + 64)
    c = np.zeros(n, dtype=np.int64)
    used, off, placed = set(), 0, 0
    while placed < TAU:
        if off + 5 > len(raw):
            raw = shake_256(raw).digest(16 * TAU + 64)
            off = 0
        idx = int.from_bytes(raw[off:off + 4], "little") % n
        sgn = 1 if raw[off + 4] & 1 else -1
        off += 5
        if idx in used:
            continue
        used.add(idx)
        c[idx] = sgn
        placed += 1
    return ring.small_int(c), c


def hash_state(*parts):
    h = shake_256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        elif isinstance(p, int):
            h.update(int(p).to_bytes(8, "little", signed=True))
        else:
            h.update(np.ascontiguousarray(np.asarray(p, dtype=np.int64)).tobytes())
    return h.digest(32)


def rejection_accept(z_raw, shift_raw, sigma, rng, reject_m=1.25):
    num = float(np.dot(shift_raw.astype(float), shift_raw.astype(float)))
    dot = float(np.dot(z_raw.astype(float), shift_raw.astype(float)))
    arg = (-2.0 * dot + num) / (2.0 * sigma * sigma)
    val = np.exp(min(arg, 700.0)) / reject_m
    return rng.random() < min(1.0, val)


def proof_widths(ring, n_c, sigma_err=3.2):
    """Masking widths for the voter and for the ballot box, calibrated so that
    one full casting round succeeds with probability about one half."""
    reject_m = 2.0 ** (1.0 / (n_c + 1))
    alpha = 12.0 / np.log(reject_m)
    s1 = alpha * (TAU ** 0.5) * sigma_err * (3 * ring.n) ** 0.5
    s2 = alpha * (s1 * (3 * ring.n) ** 0.5 + (TAU ** 0.5) * sigma_err * (3 * ring.n) ** 0.5)
    return s1, s2, reject_m


class BallotProof:
    """Sigma-OR proof that a Regev ciphertext encrypts one of n_C public
    plaintexts. The Fiat-Shamir challenge is bound to the first messages only,
    which is what lets the ballot box re-randomise the statement and adapt the
    proof without the witness."""

    def __init__(self, pke, n_c, sigma_mask, sigma_launder, reject_m=None):
        self.E = pke
        self.R = pke.R
        self.n_c = n_c
        self.sigma_mask = sigma_mask
        self.sigma_launder = sigma_launder
        self.reject_m = reject_m if reject_m is not None else 2.0 ** (1.0 / (n_c + 1))
        self.bound = int(np.ceil(6.0 * sigma_launder))

    def shift(self, k):
        R = self.R
        vec = np.zeros(R.n, dtype=np.int64)
        vec[k] = 1
        return np.stack([R.zeros(), R.scale(vec, self.E.delta)])

    def linear_map(self, pk, w):
        R = self.R
        a, b = pk
        r, f0, f1 = w
        return np.stack([R.add(R.mul(a, r), f0), R.add(R.mul(b, r), f1)])

    def _gauss3(self, rng, sigma):
        R = self.R
        res, raw = [], []
        for _ in range(3):
            v = np.rint(rng.normal(0.0, sigma, size=R.n)).astype(np.int64)
            res.append(R.small_int(v))
            raw.append(v)
        return res, np.concatenate(raw)

    def _to_raw(self, w):
        return np.concatenate([self.R.center_int(x).astype(np.int64) for x in w])

    def commit(self, pk, rng, ct, k):
        R = self.R
        seeds, zs, alphas, raws = [], [], [], []
        for j in range(self.n_c):
            if j == k:
                y, y_raw = self._gauss3(rng, self.sigma_mask)
                alphas.append(self.linear_map(pk, y))
                seeds.append(None)
                zs.append(y)
                raws.append(y_raw)
            else:
                sj = bytes(rng.integers(0, 256, size=32, dtype=np.int64).astype(np.uint8))
                cj, _ = sample_challenge(R, sj)
                zj, zj_raw = self._gauss3(rng, self.sigma_mask)
                stmt = R.sub(ct, self.shift(j))
                aj = R.sub(self.linear_map(pk, zj), R.mul(cj, stmt))
                alphas.append(aj)
                seeds.append(sj)
                zs.append(zj)
                raws.append(zj_raw)
        return alphas, seeds, zs, raws

    def launder_commit(self, pk, rng, ct, alphas):
        R = self.R
        rz, rz_raw = self._gauss3(rng, self.E.sigma)
        ct2 = R.add(ct, self.linear_map(pk, rz))
        masks, mask_raws, alphas2 = [], [], []
        for j in range(self.n_c):
            y, y_raw = self._gauss3(rng, self.sigma_launder)
            masks.append(y)
            mask_raws.append(y_raw)
            alphas2.append(R.add(alphas[j], self.linear_map(pk, y)))
        return ct2, alphas2, (rz, rz_raw, masks, mask_raws)

    def challenge(self, ctx, ct, alphas2):
        return hash_state(ctx, ct, *list(alphas2))

    def respond(self, pk, rng, k, witness, wit_raw, seeds, zs, raws, ctx, ct2, alphas2):
        R = self.R
        digest = self.challenge(ctx, ct2, alphas2)
        acc = int.from_bytes(digest, "little")
        for j in range(self.n_c):
            if j != k:
                acc ^= int.from_bytes(seeds[j], "little")
        sk_seed = acc.to_bytes(32, "little")
        ck, ck_raw = sample_challenge(R, sk_seed)
        zk = [R.add(zs[k][i], R.mul(ck, witness[i])) for i in range(3)]
        shift_raw = np.concatenate(
            [self.R.center_int(R.mul(ck, witness[i])).astype(np.int64) for i in range(3)]
        )
        zk_raw = self._to_raw(zk)
        if not rejection_accept(zk_raw, shift_raw, self.sigma_mask, rng, self.reject_m):
            return None
        out_seeds, out_zs, out_raws = list(seeds), list(zs), list(raws)
        out_seeds[k] = sk_seed
        out_zs[k] = zk
        out_raws[k] = zk_raw
        return out_seeds, out_zs, out_raws

    def finalize(self, rng, seeds, zs, raws, aux):
        R = self.R
        rz, rz_raw, masks, mask_raws = aux
        out, out_raw = [], []
        for j in range(self.n_c):
            cj, _ = sample_challenge(R, seeds[j])
            cr = [R.mul(cj, rz[i]) for i in range(3)]
            shift = raws[j] + np.concatenate(
                [R.center_int(cr[i]).astype(np.int64) for i in range(3)]
            )
            zj = [R.add(R.add(zs[j][i], masks[j][i]), cr[i]) for i in range(3)]
            zj_raw = self._to_raw(zj)
            if not rejection_accept(zj_raw, shift, self.sigma_launder, rng, self.reject_m):
                return None
            out.append(zj)
            out_raw.append(zj_raw)
        return out, out_raw

    def verify(self, pk, ct, ctx, alphas, seeds, zs):
        R = self.R
        for j in range(self.n_c):
            cj, _ = sample_challenge(R, seeds[j])
            stmt = R.sub(ct, self.shift(j))
            lhs = self.linear_map(pk, zs[j])
            rhs = R.add(alphas[j], R.mul(cj, stmt))
            if not np.array_equal(lhs, rhs):
                return False
            for i in range(3):
                if R.infty(zs[j][i]) > self.bound:
                    return False
        digest = self.challenge(ctx, ct, alphas)
        acc = int.from_bytes(digest, "little")
        for j in range(self.n_c):
            acc ^= int.from_bytes(seeds[j], "little")
        return acc == 0

    def proof_bytes(self):
        coeff_bits = int(np.ceil(np.log2(2 * self.bound + 1)))
        resp = self.n_c * 3 * self.R.n * coeff_bits / 8.0
        return int(resp + 32 * self.n_c)
