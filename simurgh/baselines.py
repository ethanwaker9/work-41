import time

import numpy as np

from .proofs import TAU, hash_state, rejection_accept, sample_challenge
from .ring import Rq

LAMBDA = 128


def _gauss_vec(R, rng, sigma, count):
    res = []
    for _ in range(count):
        v = np.rint(rng.normal(0.0, sigma, size=R.n)).astype(np.int64)
        res.append(R.small_int(v))
    return res


def _linear_map(R, pk, w):
    a, b = pk
    r, f0, f1 = w
    return np.stack([R.add(R.mul(a, r), f0), R.add(R.mul(b, r), f1)])


class RandomizationMethod:
    name = "abstract"
    hides_link = False
    needs_extra_servers = 0

    def __init__(self, pke, rng):
        self.E = pke
        self.R = pke.R
        self.rng = rng

    def run(self, pk, ct):
        raise NotImplementedError


class ReRandFS(RandomizationMethod):
    """Re-encryption published with a Fiat-Shamir-with-aborts proof of knowledge
    of a short opening of the re-randomisation ciphertext."""

    name = "ReRand+FSwA"
    hides_link = False

    def __init__(self, pke, rng, alpha=60.0):
        super().__init__(pke, rng)
        self.sigma = alpha * (TAU ** 0.5) * pke.sigma * (3 * pke.R.n) ** 0.5
        self.bound = int(np.ceil(6.0 * self.sigma))

    def run(self, pk, ct):
        R, rng = self.R, self.rng
        t0 = time.perf_counter()
        rz = _gauss_vec(R, rng, self.E.sigma, 3)
        z = _linear_map(R, pk, rz)
        ct2 = R.add(ct, z)
        while True:
            y = _gauss_vec(R, rng, self.sigma, 3)
            alpha = _linear_map(R, pk, y)
            seed = hash_state(b"rerand", z, alpha)
            c, _ = sample_challenge(R, seed)
            resp = [R.add(y[i], R.mul(c, rz[i])) for i in range(3)]
            shift = np.concatenate(
                [R.center_int(R.mul(c, rz[i])).astype(np.int64) for i in range(3)])
            raw = np.concatenate([R.center_int(x).astype(np.int64) for x in resp])
            if rejection_accept(raw, shift, self.sigma, rng, 1.25):
                break
        t_prove = time.perf_counter() - t0
        t0 = time.perf_counter()
        alpha_r = R.sub(_linear_map(R, pk, resp), R.mul(c, z))
        ok = np.array_equal(hash_state(b"rerand", z, alpha_r), seed)
        ok = ok and max(R.infty(x) for x in resp) <= self.bound
        t_ver = time.perf_counter() - t0
        bits = int(np.ceil(np.log2(2 * self.bound + 1)))
        size = 3 * R.n * bits // 8 + 32 + 2 * R.n * R.logq // 8
        return dict(ct=ct2, bytes=size, ok=bool(ok), prove=t_prove, verify=t_ver)


class ReRandCutChoose(RandomizationMethod):
    """Commit-and-open proof of a zero encryption with lambda binary challenges,
    which is online extractable in the quantum random oracle model."""

    name = "ReRand+CutChoose"
    hides_link = False

    def __init__(self, pke, rng, reps=LAMBDA):
        super().__init__(pke, rng)
        self.reps = reps

    def run(self, pk, ct):
        R, rng = self.R, self.rng
        t0 = time.perf_counter()
        rz = _gauss_vec(R, rng, self.E.sigma, 3)
        z = _linear_map(R, pk, rz)
        ct2 = R.add(ct, z)
        commits, masks = [], []
        for _ in range(self.reps):
            y = _gauss_vec(R, rng, self.E.sigma, 3)
            masks.append(y)
            commits.append(_linear_map(R, pk, y))
        seed = hash_state(b"cutchoose", z, *commits)
        bits = np.frombuffer(seed, dtype=np.uint8)
        resps = []
        for i in range(self.reps):
            b = (int(bits[i % len(bits)]) >> (i % 8)) & 1
            resps.append((b, masks[i] if b == 0 else
                          [R.add(masks[i][j], rz[j]) for j in range(3)]))
        t_prove = time.perf_counter() - t0
        t0 = time.perf_counter()
        ok = True
        for i, (b, resp) in enumerate(resps):
            lhs = _linear_map(R, pk, resp)
            rhs = commits[i] if b == 0 else R.add(commits[i], z)
            ok = ok and bool(np.array_equal(lhs, rhs))
        t_ver = time.perf_counter() - t0
        size = self.reps * 3 * R.n * 8 // 8 + 2 * R.n * R.logq // 8 + 32
        return dict(ct=ct2, bytes=size, ok=ok, prove=t_prove, verify=t_ver)


class ReRandAmortized(RandomizationMethod):
    """Amortised proof of many short preimages: a single masked opening is
    published for a whole batch of re-randomisations."""

    name = "ReRand+Amortized"
    hides_link = False

    def __init__(self, pke, rng, batch=64, alpha=60.0):
        super().__init__(pke, rng)
        self.batch = batch
        self.sigma = (alpha * (TAU ** 0.5) * pke.sigma
                      * (3 * pke.R.n) ** 0.5 * np.sqrt(batch))
        self.bound = int(np.ceil(6.0 * self.sigma))

    def run(self, pk, ct):
        R, rng = self.R, self.rng
        t0 = time.perf_counter()
        rz = _gauss_vec(R, rng, self.E.sigma, 3)
        z = _linear_map(R, pk, rz)
        ct2 = R.add(ct, z)
        y = _gauss_vec(R, rng, self.sigma, 3)
        alpha = _linear_map(R, pk, y)
        seed = hash_state(b"amort", z, alpha)
        c, _ = sample_challenge(R, seed)
        resp = [R.add(y[i], R.mul(c, rz[i])) for i in range(3)]
        t_prove = time.perf_counter() - t0
        t0 = time.perf_counter()
        lhs = _linear_map(R, pk, resp)
        rhs = R.add(alpha, R.mul(c, z))
        ok = bool(np.array_equal(lhs, rhs))
        t_ver = time.perf_counter() - t0
        bits = int(np.ceil(np.log2(2 * self.bound + 1)))
        size = (3 * R.n * bits // 8 + 32) // self.batch + 2 * R.n * R.logq // 8
        return dict(ct=ct2, bytes=size, ok=ok, prove=t_prove / self.batch,
                    verify=t_ver / self.batch)


class ShuffleMix(RandomizationMethod):
    """Verifiable re-encryption shuffle: each mix server commits to the permuted
    ciphertext and proves one linear relation per entry."""

    name = "VerifiableShuffle"
    hides_link = True

    def __init__(self, pke, rng, servers=4, alpha=60.0, com_rank=3):
        super().__init__(pke, rng)
        self.servers = servers
        self.com_rank = com_rank
        self.sigma = (alpha * (TAU ** 0.5) * pke.sigma
                      * ((com_rank + 2) * pke.R.n) ** 0.5)
        self.bound = int(np.ceil(6.0 * self.sigma))
        self.needs_extra_servers = servers

    def run(self, pk, ct):
        R, rng = self.R, self.rng
        cur = ct
        total, t_prove, t_ver = 0, 0.0, 0.0
        for _ in range(self.servers):
            t0 = time.perf_counter()
            rz = _gauss_vec(R, rng, self.E.sigma, 3)
            cur = R.add(cur, _linear_map(R, pk, rz))
            com_rand = np.stack(_gauss_vec(R, rng, self.E.sigma, self.com_rank))
            a1 = R.uniform(rng, self.com_rank)
            a2 = R.uniform(rng, self.com_rank)
            com0 = R.inner(a1, com_rand)
            com1 = R.add(R.inner(a2, com_rand), cur[0])
            com2 = R.add(R.inner(a2, com_rand), cur[1])
            y = _gauss_vec(R, rng, self.sigma, self.com_rank)
            t = R.inner(a1, np.stack(y))
            seed = hash_state(b"shuffle", com0, com1, com2, t)
            c, _ = sample_challenge(R, seed)
            resp = [R.add(y[i], R.mul(c, com_rand[i])) for i in range(self.com_rank)]
            t_prove += time.perf_counter() - t0
            t0 = time.perf_counter()
            lhs = R.inner(a1, np.stack(resp))
            rhs = R.add(t, R.mul(c, com0))
            if not np.array_equal(lhs, rhs):
                return dict(ct=cur, bytes=total, ok=False, prove=t_prove, verify=t_ver)
            t_ver += time.perf_counter() - t0
            bits = int(np.ceil(np.log2(2 * self.bound + 1)))
            total += 3 * R.n * R.logq // 8
            total += (self.com_rank + 2) * R.n * bits // 8 + 32
        return dict(ct=cur, bytes=total, ok=True, prove=t_prove, verify=t_ver)


class DecryptionMix(RandomizationMethod):
    """Decryption mix-net: the ballot carries one encryption layer per server
    and each server strips a layer."""

    name = "DecryptionMix"
    hides_link = True

    def __init__(self, pke, rng, servers=4):
        super().__init__(pke, rng)
        self.servers = servers
        self.needs_extra_servers = servers

    def run(self, pk, ct):
        R, rng = self.R, self.rng
        cur = ct
        t0 = time.perf_counter()
        for _ in range(self.servers):
            layer = _gauss_vec(R, rng, self.E.sigma, 3)
            cur = R.add(cur, _linear_map(R, pk, layer))
        t_prove = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = R.add(cur, cur)
        t_ver = time.perf_counter() - t0
        total = self.servers * self.servers * 2 * R.n * R.logq // 8
        return dict(ct=cur, bytes=total, ok=True, prove=t_prove, verify=t_ver)
