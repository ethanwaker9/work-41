import argparse
import csv
import json
import os
import secrets
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simurgh.classical import (ElGamal, chide_cleansing_ops, jcj_cleansing_ops,
                               modexp_cost)
from simurgh.fahe import HomPKE
from simurgh.params import find_prime
from simurgh.ring import Rq, ntt_primes
from simurgh.weights import BLIND_BITS, CHAL_BITS, REPS

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SIZES = [10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6]


def measure_pet(reps=6):
    eg = ElGamal()
    a = eg.enc(7)
    b = eg.enc(7)
    t0 = time.perf_counter()
    for _ in range(reps):
        eg.pet(a, b)
    return (time.perf_counter() - t0) / reps


def measure_gate(reps=6):
    """One conditional gate of the sorting based cleansing: a blinded
    re-encryption followed by a threshold decryption."""
    eg = ElGamal()
    a = eg.enc(7)
    t0 = time.perf_counter()
    for _ in range(reps):
        z = secrets.randbits(256) + 1
        eg.dec(eg.exp(eg.mul(a, eg.enc(1)), z))
    return (time.perf_counter() - t0) / reps


def measure_simurgh(degree=4096, limbs=3, bits=28, slots=64, trustees=4, reps=3):
    rng = np.random.default_rng(2)
    R = Rq(degree, ntt_primes(degree, limbs, bits))
    E = HomPKE(R, find_prime(1 << 17, 8, 3), 3.2)
    pk, _ = E.keygen(rng)
    msg = np.zeros(R.n, dtype=np.int64)
    msg[0] = 1
    cts = [E.encrypt(pk, rng, msg)[0] for _ in range(8)]
    weights = [int(rng.integers(0, E.p)) for _ in range(trustees)]
    t0 = time.perf_counter()
    for _ in range(reps):
        acc = R.zeros(2)
        for j in range(slots):
            ct = cts[j % len(cts)]
            for w in weights:
                acc = R.add(acc, R.mul_scalar_int(ct, w))
    per_slot = (time.perf_counter() - t0) / (reps * slots)
    return per_slot, 2 * R.n * R.logq // 8, trustees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    args = ap.parse_args()

    exp_ms = modexp_cost(20) * 1000
    pet_ms = measure_pet(args.reps) * 1000
    gate_ms = measure_gate(args.reps) * 1000
    slot_ms, ct_bytes, trustees = measure_simurgh(reps=args.reps)
    slot_ms *= 1000

    rows = []
    for n in SIZES:
        n_b = 2 * n
        jcj = jcj_cleansing_ops(n_b, n)
        chide = chide_cleansing_ops(n_b, n)
        simurgh_ops = 2 * n * trustees
        rows.append({
            "voters": n,
            "jcj_ops": jcj["exps"],
            "chide_ops": chide["exps"],
            "simurgh_ops": simurgh_ops,
            "jcj_seconds": jcj["pets"] * pet_ms / 1000.0,
            "chide_seconds": chide["gates"] * gate_ms / 1000.0,
            "simurgh_seconds": 2 * n * slot_ms / 1000.0,
            "jcj_bytes": jcj["pets"] * 768,
            "chide_bytes": chide["gates"] * 768,
            "simurgh_bytes": trustees * (ct_bytes
                                         + REPS * 2 * n * (BLIND_BITS + CHAL_BITS + 2) // 8
                                         + REPS * 6 * 4096 * 84 // 8),
        })

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "tally.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {"modexp_ms": exp_ms, "pet_ms": pet_ms, "gate_ms": gate_ms,
            "simurgh_slot_ms": slot_ms, "ciphertext_bytes": ct_bytes,
            "trustees": trustees}
    with open(os.path.join(RESULTS, "tally_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    for r in rows:
        print("%9d  JCJ %.3g s / %.3g GB   CHide %.3g s / %.3g GB   Simurgh %.3g s / %.3g GB" % (
            r["voters"], r["jcj_seconds"], r["jcj_bytes"] / 1e9,
            r["chide_seconds"], r["chide_bytes"] / 1e9,
            r["simurgh_seconds"], r["simurgh_bytes"] / 1e9))


if __name__ == "__main__":
    main()
