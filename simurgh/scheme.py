import numpy as np
from dataclasses import dataclass, field
from hashlib import shake_256

from dilithium_py.ml_dsa import ML_DSA_65

from .dkg import ThresholdKey, renyi_flooding_sigma
from .fahe import AnamorphicPKE, HomPKE
from .proofs import BallotProof, TAU, hash_state, proof_widths
from .ring import Rq, ntt_primes
from .weights import WeightProof
from .params import find_prime


@dataclass
class PublicData:
    pk: object
    roster: list
    bit_commit: list
    n_c: int
    eid: bytes
    reg_pk: bytes


@dataclass
class Ballot:
    slot: int
    seq: int
    ct: object
    seeds: list
    zs: list
    sig: bytes


class Simurgh:
    def __init__(self, n_c=4, n_voters=16, n_trustees=4, threshold=3,
                 ring_degree=4096, ring_limbs=3, limb_bits=28,
                 caps_degree=2048, caps_bits=30, alpha=60.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.R = Rq(ring_degree, ntt_primes(ring_degree, ring_limbs, limb_bits))
        self.p = find_prime(1 << 17, 8, 3)
        self.E = HomPKE(self.R, self.p, 3.2)
        self.n_c = n_c
        self.n_voters = n_voters
        self.n_slots = 2 * n_voters
        self.n_trustees = n_trustees
        self.threshold = threshold
        sigma1, sigma2, reject_m = proof_widths(self.R, n_c)
        self.proof = BallotProof(self.E, n_c, sigma1, sigma2, reject_m)
        self.Rc = Rq(caps_degree, ntt_primes(caps_degree, 1, caps_bits))
        self.A = AnamorphicPKE(self.Rc, p=251, p_hat=251, sigma=3.2,
                               sigma_trap=256.0, m_bar=5)
        self.eid = b"simurgh-election-2026"

    def setup(self):
        rng = self.rng
        self.pk, sk = self.E.keygen(rng)
        s_raw = self.R.center_int(sk[0]).astype(object) % self.R.q
        noise_bound = 1 << 40
        self.tkey = ThresholdKey(self.R, self.n_trustees, self.threshold,
                                 renyi_flooding_sigma(noise_bound, 1 << 16))
        self.shares = self.tkey.share(rng, s_raw)
        self.sk = sk
        self.reg_pk, self.reg_sk = ML_DSA_65.keygen()
        self.slot_keys = [ML_DSA_65.keygen() for _ in range(self.n_slots)]
        perm = rng.permutation(self.n_slots)
        self.real_slot = {i: int(perm[2 * i]) for i in range(self.n_voters)}
        self.decoy_slot = {i: int(perm[2 * i + 1]) for i in range(self.n_voters)}
        self.realness = np.zeros(self.n_slots, dtype=np.int64)
        for i in range(self.n_voters):
            self.realness[self.real_slot[i]] = 1
        self.bit_shares = []
        for j in range(self.n_slots):
            sh = [int(x) for x in rng.integers(0, self.p, size=self.n_trustees - 1)]
            sh.append((int(self.realness[j]) - sum(sh)) % self.p)
            self.bit_shares.append(sh)
        self.roster = [kp[0] for kp in self.slot_keys]
        self.wproof = WeightProof(self.R, self.n_slots, rng)
        self.weight_com = [self.wproof.commit([self.bit_shares[j][k]
                                               for j in range(self.n_slots)])
                           for k in range(self.n_trustees)]
        self.board = {}
        for j in range(self.n_slots):
            null = np.zeros(self.R.n, dtype=np.int64)
            ct, _ = self.E.encrypt(self.pk, rng, null)
            self.board[j] = Ballot(j, 0, ct, [], [], b"")
        return PublicData(self.pk, self.roster, self.bit_shares, self.n_c,
                          self.eid, self.reg_pk)

    def register(self, voter):
        rng = self.rng
        apk, ask, trap = self.A.agen(rng)
        real = self._slot_payload(self.real_slot[voter])
        decoy = self._slot_payload(self.decoy_slot[voter])
        capsule = self.A.aencrypt(apk, rng, decoy, real)
        return {"apk": apk, "ask": ask, "tk": trap, "capsule": capsule}

    def _slot_payload(self, slot):
        digits = []
        s = slot
        for _ in range(4):
            digits.append(s % 251)
            s //= 251
        return self.A.encode(digits, 251)

    def _payload_slot(self, digits):
        return sum(int(d) * (251 ** i) for i, d in enumerate(digits[:4]))

    def open_real(self, cred):
        return self._payload_slot(self.A.adecrypt(cred["tk"], cred["capsule"], 4))

    def fake_cred(self, cred):
        return self._payload_slot(self.A.decrypt(cred["ask"], cred["capsule"], 4))

    def vote(self, slot, choice, seq, max_attempts=200):
        rng = self.rng
        msg = np.zeros(self.R.n, dtype=np.int64)
        msg[choice] = 1
        ct, wit = self.E.encrypt(self.pk, rng, msg)
        ctx = hash_state(self.eid, slot, seq)
        for _ in range(max_attempts):
            alphas, seeds, zs, raws = self.proof.commit(self.pk, rng, ct, choice)
            ct2, alphas2, aux = self.proof.launder_commit(self.pk, rng, ct, alphas)
            r = self.proof.respond(self.pk, rng, choice, wit, None, seeds, zs,
                                   raws, ctx, ct2, alphas2)
            if r is None:
                continue
            seeds2, zs2, raws2 = r
            fin = self.proof.finalize(rng, seeds2, zs2, raws2, aux)
            if fin is None:
                continue
            zs3, _ = fin
            sk = self.slot_keys[slot][1]
            payload = ctx + np.ascontiguousarray(ct2).tobytes() + b"".join(seeds2)
            sig = ML_DSA_65.sign(sk, shake_256(payload).digest(64))
            return Ballot(slot, seq, ct2, seeds2, zs3, sig)
        raise RuntimeError("ballot casting did not terminate")

    def valid(self, ballot):
        ctx = hash_state(self.eid, ballot.slot, ballot.seq)
        ok = self.verify_proof(ballot)
        if not ok:
            return False
        payload = ctx + np.ascontiguousarray(ballot.ct).tobytes() + b"".join(ballot.seeds)
        return ML_DSA_65.verify(self.roster[ballot.slot],
                                shake_256(payload).digest(64), ballot.sig)

    def verify_proof(self, ballot):
        R = self.R
        ctx = hash_state(self.eid, ballot.slot, ballot.seq)
        alphas = []
        for j in range(self.n_c):
            from .proofs import sample_challenge
            cj, _ = sample_challenge(R, ballot.seeds[j])
            stmt = R.sub(ballot.ct, self.proof.shift(j))
            alphas.append(R.sub(self.proof.linear_map(self.pk, ballot.zs[j]),
                                R.mul(cj, stmt)))
        return self.proof.verify(self.pk, ballot.ct, ctx, alphas, ballot.seeds,
                                 ballot.zs)

    def cast(self, ballot):
        if not self.valid(ballot):
            return False
        cur = self.board.get(ballot.slot)
        if cur is not None and cur.seq >= ballot.seq:
            return False
        self.board[ballot.slot] = ballot
        return True

    def tally(self, index_set=None):
        rng = self.rng
        R = self.R
        index_set = index_set or list(range(1, self.threshold + 1))
        slots = sorted(self.board.keys())
        cts = [self.board[j].ct for j in slots]
        partial_sums, wproofs = [], []
        for k in range(self.n_trustees):
            shares = [self.bit_shares[j][k] for j in slots]
            acc = R.zeros(2)
            for ct, w in zip(cts, shares):
                acc = R.add(acc, R.mul_scalar_int(ct, w))
            partial_sums.append(acc)
            wproofs.append(self.wproof.prove(shares, cts, acc, b"weights"))
        agg = partial_sums[0]
        for k in range(1, self.n_trustees):
            agg = R.add(agg, partial_sums[k])
        parts = [self.tkey.partial_decrypt(self.shares[i - 1], agg[0], rng,
                                           index_set, i) for i in index_set]
        raw = self.tkey.combine(agg[1], parts)
        result = self.E.decode(R.to_int(raw), self.n_c)
        return result, {"agg": agg, "partials": partial_sums,
                        "wproofs": wproofs, "slots": slots}

    def verify_tally(self, result, transcript):
        R = self.R
        slots = transcript["slots"]
        cts = [self.board[j].ct for j in slots]
        acc = R.zeros(2)
        for k in range(self.n_trustees):
            if not self.wproof.verify(self.weight_com[k], cts,
                                      transcript["partials"][k], b"weights",
                                      transcript["wproofs"][k]):
                return False
            acc = R.add(acc, transcript["partials"][k])
        return bool(np.array_equal(acc, transcript["agg"]))

    def transcript_bytes(self):
        ct_bytes = 2 * self.R.n * self.R.logq // 8
        return self.n_trustees * (ct_bytes + self.wproof.size_bytes())
