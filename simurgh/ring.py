import numpy as np
from hashlib import shake_256


def _is_prime(x):
    if x < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if x % p == 0:
            return x == p
    d, r = x - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        y = pow(a, d, x)
        if y in (1, x - 1):
            continue
        for _ in range(r - 1):
            y = y * y % x
            if y == x - 1:
                break
        else:
            return False
    return True


def ntt_primes(n, count, start_bits):
    step = 2 * n
    out = []
    cand = ((1 << start_bits) // step) * step + 1
    while len(out) < count:
        if _is_prime(cand):
            out.append(cand)
        cand -= step
    return out


def _primitive_root(p):
    fac, x = [], p - 1
    d = 2
    while d * d <= x:
        if x % d == 0:
            fac.append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        fac.append(x)
    g = 2
    while True:
        if all(pow(g, (p - 1) // f, p) != 1 for f in fac):
            return g
        g += 1


def _bitrev(i, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


class NTT:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.bits = n.bit_length() - 1
        g = _primitive_root(p)
        psi = pow(g, (p - 1) // (2 * n), p)
        psi_inv = pow(psi, p - 2, p)
        self.n_inv = pow(n, p - 2, p)
        self.psi_rev = np.array([pow(psi, _bitrev(i, self.bits), p) for i in range(n)], dtype=np.int64)
        self.psi_inv_rev = np.array([pow(psi_inv, _bitrev(i, self.bits), p) for i in range(n)], dtype=np.int64)

    def forward(self, a):
        p, n = self.p, self.n
        a = np.array(a, dtype=np.int64, copy=True)
        lead = a.shape[:-1]
        t = n
        m = 1
        while m < n:
            t //= 2
            v = a.reshape(lead + (m, 2, t))
            s = self.psi_rev[m:2 * m].reshape((m, 1))
            hi = (v[..., 1, :] * s) % p
            lo = v[..., 0, :].copy()
            v[..., 0, :] = (lo + hi) % p
            v[..., 1, :] = (lo - hi) % p
            a = v.reshape(lead + (n,))
            m *= 2
        return a

    def inverse(self, a):
        p, n = self.p, self.n
        a = np.array(a, dtype=np.int64, copy=True)
        lead = a.shape[:-1]
        t = 1
        m = n
        while m > 1:
            h = m // 2
            v = a.reshape(lead + (h, 2, t))
            s = self.psi_inv_rev[h:2 * h].reshape((h, 1))
            lo = v[..., 0, :].copy()
            hi = v[..., 1, :].copy()
            v[..., 0, :] = (lo + hi) % p
            v[..., 1, :] = ((lo - hi) * s) % p
            a = v.reshape(lead + (n,))
            t *= 2
            m = h
        return (a * self.n_inv) % p


class Rq:
    """Cyclotomic ring Z[X]/(X^n+1) modulo q, stored in RNS residue form."""

    def __init__(self, n, primes):
        self.n = n
        self.primes = list(primes)
        self.L = len(primes)
        self.q = 1
        for p in primes:
            self.q *= p
        self.ntts = [NTT(n, p) for p in primes]
        self.pa = np.array(self.primes, dtype=np.int64).reshape((self.L, 1))
        self.crt_coeff = []
        for p in primes:
            qi = self.q // p
            self.crt_coeff.append(qi * pow(qi % p, p - 2, p))
        self.logq = self.q.bit_length()

    def shape_of(self, *dims):
        return tuple(dims) + (self.L, self.n)

    def zeros(self, *dims):
        return np.zeros(self.shape_of(*dims), dtype=np.int64)

    def _pmod(self, a):
        return a % self.pa

    def add(self, a, b):
        return (a + b) % self.pa

    def sub(self, a, b):
        return (a - b) % self.pa

    def neg(self, a):
        return (-a) % self.pa

    def mul(self, a, b):
        a = np.broadcast_to(a, np.broadcast_shapes(a.shape, b.shape))
        b = np.broadcast_to(b, a.shape)
        out = np.empty(a.shape, dtype=np.int64)
        for i, nt in enumerate(self.ntts):
            fa = nt.forward(a[..., i, :])
            fb = nt.forward(b[..., i, :])
            out[..., i, :] = nt.inverse((fa * fb) % nt.p)
        return out

    def mul_scalar_int(self, a, k):
        k = np.array([int(k) % p for p in self.primes], dtype=np.int64).reshape((self.L, 1))
        return (a * k) % self.pa

    def matvec(self, mat, vec):
        rows = mat.shape[0]
        out = self.zeros(rows)
        for i in range(rows):
            out[i] = self.inner(mat[i], vec)
        return out

    def inner(self, u, v):
        prod = self.mul(u, v)
        return prod.sum(axis=0) % self.pa

    def from_int(self, arr):
        arr = np.asarray(arr, dtype=object)
        out = np.empty(arr.shape + (self.L, self.n), dtype=np.int64)
        flat = arr.reshape(-1, self.n)
        res = np.empty((flat.shape[0], self.L, self.n), dtype=np.int64)
        for i, p in enumerate(self.primes):
            res[:, i, :] = np.array([[int(x) % p for x in row] for row in flat], dtype=np.int64)
        return res.reshape(arr.shape[:-1] + (self.L, self.n))

    def to_int(self, a):
        lead = a.shape[:-2]
        flat = a.reshape(-1, self.L, self.n)
        out = np.zeros((flat.shape[0], self.n), dtype=object)
        for i in range(self.L):
            out = out + flat[:, i, :].astype(object) * self.crt_coeff[i]
        q = self.q
        out = np.vectorize(lambda x: int(x) % q, otypes=[object])(out)
        return out.reshape(lead + (self.n,))

    def center_int(self, a):
        v = self.to_int(a)
        q, h = self.q, self.q // 2
        return np.vectorize(lambda x: int(x) - q if int(x) > h else int(x), otypes=[object])(v)

    def infty(self, a):
        c = self.center_int(a)
        if c.size == 0:
            return 0
        return int(max(abs(int(x)) for x in c.reshape(-1)))

    def uniform(self, rng, *dims):
        out = np.empty(self.shape_of(*dims), dtype=np.int64)
        for i, p in enumerate(self.primes):
            out[..., i, :] = rng.integers(0, p, size=tuple(dims) + (self.n,), dtype=np.int64)
        return out

    def small_int(self, arr):
        arr = np.asarray(arr, dtype=np.int64)
        out = np.empty(arr.shape[:-1] + (self.L, self.n), dtype=np.int64)
        for i, p in enumerate(self.primes):
            out[..., i, :] = arr % p
        return out

    def scale(self, arr, scalar):
        arr = np.asarray(arr, dtype=np.int64)
        out = np.empty(arr.shape[:-1] + (self.L, self.n), dtype=np.int64)
        for i, pr in enumerate(self.primes):
            sp = int(scalar) % pr
            out[..., i, :] = ((arr % pr) * sp) % pr
        return out

    def ternary(self, rng, *dims):
        raw = rng.integers(-1, 2, size=tuple(dims) + (self.n,), dtype=np.int64)
        return self.small_int(raw), raw

    def gaussian(self, rng, sigma, *dims):
        raw = np.rint(rng.normal(0.0, sigma, size=tuple(dims) + (self.n,))).astype(np.int64)
        return self.small_int(raw), raw

    def expand(self, seed, *dims):
        total = 1
        for d in dims:
            total *= d
        out = np.empty(self.shape_of(*dims), dtype=np.int64)
        nbytes = 4
        raw = shake_256(seed).digest(total * self.L * self.n * nbytes)
        vals = np.frombuffer(raw, dtype=np.uint32).astype(np.int64)
        vals = vals.reshape(tuple(dims) + (self.L, self.n))
        for i, p in enumerate(self.primes):
            out[..., i, :] = vals[..., i, :] % p
        return out
