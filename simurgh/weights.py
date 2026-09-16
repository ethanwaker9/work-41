import numpy as np

from .proofs import hash_state
from .ring import Rq

REPS = 8
CHAL_BITS = 16
BLIND_BITS = 56


class WeightProof:
    """Batched proof that a trustee applied its committed share vector to the
    board. The commitment is an Ajtai commitment to the packed shares and the
    relation is linear over the integers, so one scalar challenge per repetition
    binds both views of the same vector."""

    def __init__(self, ring: Rq, n_slots, rng, com_rank=4):
        self.R = ring
        self.n_slots = n_slots
        self.rng = rng
        self.com_rank = com_rank
        self.blocks = int(np.ceil(n_slots / ring.n))
        self.A = ring.uniform(rng, com_rank, self.blocks)

    def pack(self, shares):
        R = self.R
        arr = np.zeros(self.blocks * R.n, dtype=np.int64)
        arr[: len(shares)] = np.asarray(shares, dtype=np.int64)
        return arr.reshape(self.blocks, R.n)

    def commit(self, shares):
        R = self.R
        packed = R.small_int(self.pack(shares))
        out = R.zeros(self.com_rank)
        for i in range(self.com_rank):
            out[i] = R.inner(self.A[i], packed)
        return out

    def prove(self, shares, cts, aggregate, tag):
        R = self.R
        beta = self.pack(shares)
        blind = self.rng.integers(-(1 << BLIND_BITS), 1 << BLIND_BITS,
                                  size=(REPS, self.blocks, R.n), dtype=np.int64)
        t1, t2 = [], []
        for rep in range(REPS):
            packed = R.small_int(blind[rep])
            c1 = R.zeros(self.com_rank)
            for i in range(self.com_rank):
                c1[i] = R.inner(self.A[i], packed)
            t1.append(c1)
            flat = blind[rep].reshape(-1)[: self.n_slots]
            acc = R.zeros(2)
            for j, ct in enumerate(cts):
                acc = R.add(acc, R.mul_scalar_int(ct, int(flat[j])))
            t2.append(acc)
        digest = hash_state(tag, aggregate, *t1, *t2)
        chal = [int.from_bytes(digest[2 * i:2 * i + 2], "little") % (1 << CHAL_BITS)
                for i in range(REPS)]
        resp = [blind[rep] + chal[rep] * beta for rep in range(REPS)]
        return {"t1": t1, "t2": t2, "chal": chal, "resp": resp}

    def size_bytes(self):
        return REPS * self.n_slots * (BLIND_BITS + CHAL_BITS + 2) // 8 + \
            REPS * (self.com_rank + 2) * self.R.n * self.R.logq // 8

    def verify(self, com, cts, aggregate, tag, proof):
        R = self.R
        digest = hash_state(tag, aggregate, *proof["t1"], *proof["t2"])
        for rep in range(REPS):
            c = int.from_bytes(digest[2 * rep:2 * rep + 2], "little") % (1 << CHAL_BITS)
            if c != proof["chal"][rep]:
                return False
            packed = R.small_int(proof["resp"][rep])
            lhs = R.zeros(self.com_rank)
            for i in range(self.com_rank):
                lhs[i] = R.inner(self.A[i], packed)
            rhs = R.add(proof["t1"][rep], R.mul_scalar_int(com, c))
            if not np.array_equal(lhs, rhs):
                return False
            flat = proof["resp"][rep].reshape(-1)[: self.n_slots]
            acc = R.zeros(2)
            for j, ct in enumerate(cts):
                acc = R.add(acc, R.mul_scalar_int(ct, int(flat[j])))
            rhs2 = R.add(proof["t2"][rep], R.mul_scalar_int(aggregate, c))
            if not np.array_equal(acc, rhs2):
                return False
        return True
