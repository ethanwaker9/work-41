import math
from dataclasses import dataclass, field

from .ring import ntt_primes, _is_prime


def gsa_delta(beta):
    if beta <= 50:
        beta = 50
    return ((math.pi * beta) ** (1.0 / beta) * beta / (2 * math.pi * math.e)) ** (
        1.0 / (2.0 * (beta - 1))
    )


def primal_usvp_beta(dim, logq, sigma):
    q = 2.0 ** logq
    best = None
    for beta in range(60, 1400, 2):
        delta = gsa_delta(beta)
        ok = False
        for m in range(max(beta, 50), 3 * dim + 200, 16):
            d = m + dim + 1
            if beta > d:
                continue
            lhs = sigma * math.sqrt(beta)
            rhs = (delta ** (2 * beta - d - 1)) * (q ** (float(m) / d))
            if lhs <= rhs:
                ok = True
                break
        if ok:
            best = beta
            break
    return best if best is not None else 1400


def core_svp_bits(beta, quantum=False):
    return (0.265 if quantum else 0.292) * beta


def security_level(dim, logq, sigma, quantum=False):
    return core_svp_bits(primal_usvp_beta(dim, logq, sigma), quantum)


def find_prime(target, cond_mod, cond_res):
    x = target | 1
    while True:
        if x % cond_mod == cond_res and _is_prime(x):
            return x
        x += 2


@dataclass
class ElectionParams:
    n: int
    primes: list
    p: int
    sigma: float
    n_c: int
    sec: int
    n_trustees: int
    threshold: int
    logq: int = field(init=False)

    def __post_init__(self):
        q = 1
        for pr in self.primes:
            q *= pr
        self.logq = q.bit_length()

    @property
    def q(self):
        q = 1
        for pr in self.primes:
            q *= pr
        return q

    def ciphertext_bytes(self):
        return 2 * self.n * self.logq // 8

    def security(self, quantum=False):
        return security_level(self.n, self.logq, self.sigma, quantum)


@dataclass
class CapsuleParams:
    n: int
    primes: list
    p: int
    sigma: float
    sigma_key: float
    m_bar: int
    ell: int
    base: int
    logq: int = field(init=False)

    def __post_init__(self):
        q = 1
        for pr in self.primes:
            q *= pr
        self.logq = q.bit_length()

    @property
    def q(self):
        q = 1
        for pr in self.primes:
            q *= pr
        return q

    @property
    def m(self):
        return self.m_bar + self.ell

    def ciphertext_bytes(self):
        return (self.m + 1) * self.n * self.logq // 8

    def security(self, quantum=False):
        return security_level(self.n, self.logq, self.sigma, quantum)


def default_election_params(n_c=8, n_trustees=4, threshold=3):
    n = 2048
    primes = ntt_primes(n, 2, 27)
    p = find_prime(1 << 17, 8, 3)
    return ElectionParams(
        n=n, primes=primes, p=p, sigma=3.2, n_c=n_c, sec=40,
        n_trustees=n_trustees, threshold=threshold,
    )


def default_capsule_params():
    n = 2048
    primes = ntt_primes(n, 1, 30)
    p = find_prime(1 << 8, 8, 3)
    return CapsuleParams(
        n=n, primes=primes, p=p, sigma=3.2, sigma_key=3.2,
        m_bar=3, ell=2, base=1 << 15,
    )
