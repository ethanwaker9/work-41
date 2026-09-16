import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simurgh.fahe import AnamorphicPKE, HomPKE
from simurgh.params import (core_svp_bits, default_capsule_params,
                            default_election_params, find_prime,
                            primal_usvp_beta)
from simurgh.proofs import BallotProof, proof_widths, sample_challenge
from simurgh.ring import Rq, ntt_primes
from simurgh.scheme import Simurgh
from simurgh.weights import WeightProof


@pytest.fixture(scope="module")
def ring():
    return Rq(1024, ntt_primes(1024, 2, 28))


def test_ring_multiplication(ring):
    rng = np.random.default_rng(0)
    a = ring.uniform(rng)
    b = ring.uniform(rng)
    c = ring.to_int(ring.mul(a, b))
    ai = ring.to_int(a).astype(object)
    bi = ring.to_int(b).astype(object)
    full = np.convolve(ai, bi)
    ref = full[: ring.n].copy()
    ref[: ring.n - 1] = ref[: ring.n - 1] - full[ring.n:]
    ref = np.array([int(x) % ring.q for x in ref], dtype=object)
    assert np.all(ref == c)


def test_homomorphic_encryption(ring):
    rng = np.random.default_rng(1)
    E = HomPKE(ring, find_prime(1 << 10, 8, 3), 3.2)
    pk, sk = E.keygen(rng)
    acc = None
    for i in range(12):
        ct, _ = E.encrypt(pk, rng, E.encode([1, 0, 1]))
        acc = ct if acc is None else E.add(acc, ct)
    assert E.decrypt(sk, acc, 3) == [12, 0, 12]


def test_anamorphic_channel():
    rng = np.random.default_rng(2)
    R = Rq(1024, ntt_primes(1024, 1, 30))
    A = AnamorphicPKE(R, p=251, p_hat=251, sigma=3.2, sigma_trap=256.0, m_bar=5)
    apk, ask, trap = A.agen(rng)
    normal = A.encode([3, 4], 251)
    covert = A.encode([200, 7], 251)
    ct = A.aencrypt(apk, rng, normal, covert)
    assert A.decrypt(ask, ct, 2) == [3, 4]
    assert A.adecrypt(trap, ct, 2) == [200, 7]
    plain = A.encrypt(apk, rng, normal)
    assert A.decrypt(ask, plain, 2) == [3, 4]
    assert A.adecrypt(trap, plain, 2, robust_bound=A.delta_hat // 4) is None
    assert A.adecrypt(trap, ct, 2, robust_bound=A.delta_hat // 4) == [200, 7]


def test_anamorphic_homomorphism():
    rng = np.random.default_rng(3)
    R = Rq(1024, ntt_primes(1024, 1, 30))
    A = AnamorphicPKE(R, p=251, p_hat=251, sigma=3.2, sigma_trap=256.0, m_bar=5)
    apk, ask, trap = A.agen(rng)
    c1 = A.aencrypt(apk, rng, A.encode([1], 251), A.encode([5], 251))
    c2 = A.aencrypt(apk, rng, A.encode([2], 251), A.encode([9], 251))
    s = A.add(c1, c2)
    assert A.decrypt(ask, s, 1) == [3]
    assert A.adecrypt(trap, s, 1) == [14]


def test_mixed_evaluation_is_rejected():
    rng = np.random.default_rng(4)
    R = Rq(1024, ntt_primes(1024, 1, 30))
    A = AnamorphicPKE(R, p=251, p_hat=251, sigma=3.2, sigma_trap=256.0, m_bar=5)
    apk, ask, trap = A.agen(rng)
    anam = A.aencrypt(apk, rng, A.encode([1], 251), A.encode([5], 251))
    plain = A.encrypt(apk, rng, A.encode([2], 251))
    mixed = A.add(anam, plain)
    assert A.adecrypt(trap, mixed, 1, robust_bound=A.delta_hat // 4) is None


def test_proof_is_randomisable_but_not_malleable():
    rng = np.random.default_rng(5)
    R = Rq(1024, ntt_primes(1024, 2, 28))
    E = HomPKE(R, find_prime(1 << 10, 8, 3), 3.2)
    pk, sk = E.keygen(rng)
    n_c = 3
    s1, s2, m = proof_widths(R, n_c)
    P = BallotProof(E, n_c, s1, s2, m)
    k = 1
    msg = np.zeros(R.n, dtype=np.int64)
    msg[k] = 1
    ct, wit = E.encrypt(pk, rng, msg)
    ctx = b"ctx"
    while True:
        alphas, seeds, zs, raws = P.commit(pk, rng, ct, k)
        ct2, alphas2, aux = P.launder_commit(pk, rng, ct, alphas)
        r = P.respond(pk, rng, k, wit, None, seeds, zs, raws, ctx, ct2, alphas2)
        if r is None:
            continue
        seeds2, zs2, raws2 = r
        fin = P.finalize(rng, seeds2, zs2, raws2, aux)
        if fin is None:
            continue
        zs3, _ = fin
        break
    assert P.verify(pk, ct2, ctx, alphas2, seeds2, zs3)
    assert E.decrypt(sk, ct2, n_c)[k] == 1
    bad = ct2.copy()
    vec = np.zeros(R.n, dtype=np.int64)
    vec[(k + 1) % n_c] = 1
    bad[1] = R.add(bad[1], R.scale(vec, E.delta))
    assert not P.verify(pk, bad, ctx, alphas2, seeds2, zs3)
    zct, zw = E.encrypt(pk, rng, np.zeros(R.n, dtype=np.int64))
    ct3 = R.add(ct2, zct)
    zs4 = []
    for j in range(n_c):
        cj, _ = sample_challenge(R, seeds2[j])
        zs4.append([R.add(zs3[j][i], R.mul(cj, zw[i])) for i in range(3)])
    assert not P.verify(pk, ct3, ctx, alphas2, seeds2, zs4)


def test_weight_proof():
    rng = np.random.default_rng(6)
    R = Rq(1024, ntt_primes(1024, 2, 28))
    E = HomPKE(R, find_prime(1 << 10, 8, 3), 3.2)
    pk, _ = E.keygen(rng)
    slots = 12
    msg = np.zeros(R.n, dtype=np.int64)
    msg[0] = 1
    cts = [E.encrypt(pk, rng, msg)[0] for _ in range(slots)]
    W = WeightProof(R, slots, rng)
    shares = [int(rng.integers(0, E.p)) for _ in range(slots)]
    com = W.commit(shares)
    agg = R.zeros(2)
    for ct, w in zip(cts, shares):
        agg = R.add(agg, R.mul_scalar_int(ct, w))
    pr = W.prove(shares, cts, agg, b"t")
    assert W.verify(com, cts, agg, b"t", pr)
    bad = R.add(agg, cts[0])
    assert not W.verify(com, cts, bad, b"t", pr)


def test_election_end_to_end():
    S = Simurgh(n_c=3, n_voters=4, seed=17)
    S.setup()
    creds = [S.register(i) for i in range(4)]
    for i, c in enumerate(creds):
        assert S.open_real(c) == S.real_slot[i]
        assert S.fake_cred(c) == S.decoy_slot[i]
    votes = [0, 1, 1, 2]
    for i, v in enumerate(votes):
        assert S.cast(S.vote(S.real_slot[i], v, seq=1))
    coerced = S.vote(S.decoy_slot[0], 2, seq=1)
    assert S.valid(coerced)
    assert S.cast(coerced)
    result, transcript = S.tally()
    assert result == [1, 2, 1]
    assert S.verify_tally(result, transcript)


def test_parameter_estimator_matches_published_levels():
    assert abs(core_svp_bits(primal_usvp_beta(512, 11.7, 1.22)) - 118.6) < 2.0
    e = default_election_params()
    c = default_capsule_params()
    assert e.p % 8 == 3
    assert c.n >= 1024
