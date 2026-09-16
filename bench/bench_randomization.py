import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simurgh.baselines import (DecryptionMix, ReRandAmortized, ReRandCutChoose,
                               ReRandFS, ShuffleMix)
from simurgh.fahe import HomPKE
from simurgh.params import find_prime
from simurgh.proofs import BallotProof, TAU, proof_widths
from simurgh.ring import Rq, ntt_primes

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def build(n_c, degree=4096, limbs=3, bits=28, alpha=60.0, seed=1):
    rng = np.random.default_rng(seed)
    R = Rq(degree, ntt_primes(degree, limbs, bits))
    E = HomPKE(R, find_prime(1 << 17, 8, 3), 3.2)
    pk, sk = E.keygen(rng)
    s1, s2, m = proof_widths(R, n_c)
    return rng, R, E, pk, sk, BallotProof(E, n_c, s1, s2, m)


def time_abl(rng, R, E, pk, proof, plain, n_c, reps):
    ctx = b"bench"
    box_t, ver_t, rounds = [], [], []
    for _ in range(reps):
        msg = np.zeros(R.n, dtype=np.int64)
        k = int(rng.integers(0, n_c))
        msg[k] = 1
        ct, wit = E.encrypt(pk, rng, msg)
        tries = 0
        while True:
            tries += 1
            alphas, seeds, zs, raws = proof.commit(pk, rng, ct, k)
            t0 = time.perf_counter()
            ct2, alphas2, aux = proof.launder_commit(pk, rng, ct, alphas)
            t_launder = time.perf_counter() - t0
            r = proof.respond(pk, rng, k, wit, None, seeds, zs, raws, ctx, ct2, alphas2)
            if r is None:
                continue
            seeds2, zs2, raws2 = r
            t0 = time.perf_counter()
            fin = proof.finalize(rng, seeds2, zs2, raws2, aux)
            t_fin = time.perf_counter() - t0
            if fin is None:
                continue
            zs3, _ = fin
            break
        box_t.append(t_launder + t_fin)
        rounds.append(tries)
        t0 = time.perf_counter()
        assert proof.verify(pk, ct2, ctx, alphas2, seeds2, zs3)
        t_laundered = time.perf_counter() - t0
        ver_t.append(t_laundered)
    return float(np.mean(box_t)), float(np.mean(ver_t)), 0, float(np.mean(rounds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n_c", type=int, default=4)
    ap.add_argument("--degree", type=int, default=4096)
    args = ap.parse_args()

    rng, R, E, pk, sk, proof = build(args.n_c, degree=args.degree)
    msg = np.zeros(R.n, dtype=np.int64)
    msg[1] = 1
    ct, _ = E.encrypt(pk, rng, msg)

    rows = []
    methods = [
        ReRandFS(E, rng),
        ReRandCutChoose(E, rng),
        ReRandAmortized(E, rng),
        ShuffleMix(E, rng, servers=4),
        DecryptionMix(E, rng, servers=4),
    ]
    for m in methods:
        ts, vs, sizes = [], [], []
        for _ in range(args.reps):
            out = m.run(pk, ct)
            assert out["ok"], m.name
            ts.append(out["prove"])
            vs.append(out["verify"])
            sizes.append(out["bytes"])
        rows.append({
            "method": m.name,
            "box_ms": 1000 * float(np.mean(ts)),
            "verify_ms": 1000 * float(np.mean(vs)),
            "total_verify_ms": 1000 * float(np.mean(vs)),
            "bytes": int(np.mean(sizes)),
            "hides_link": int(m.hides_link),
            "servers": m.needs_extra_servers,
        })

    plain = BallotProof(E, args.n_c, proof.sigma_mask, proof.sigma_mask, proof.reject_m)
    box, ver, extra, rounds = time_abl(rng, R, E, pk, proof, plain, args.n_c, args.reps)
    widening = proof.proof_bytes() - plain.proof_bytes()
    rows.append({
        "method": "ABL (this work)",
        "box_ms": 1000 * box,
        "verify_ms": 0.0,
        "total_verify_ms": 1000 * ver,
        "bytes": widening,
        "hides_link": 1,
        "servers": 0,
    })
    for r in rows[:-1]:
        r["total_verify_ms"] = r["verify_ms"] + 1000 * ver

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "randomization.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {"n_c": args.n_c, "degree": R.n, "logq": R.logq,
            "ciphertext_bytes": 2 * R.n * R.logq // 8,
            "ballot_proof_bytes": proof.proof_bytes(),
            "ballot_proof_bytes_no_laundering": plain.proof_bytes(),
            "abl_expected_rounds": rounds}
    with open(os.path.join(RESULTS, "randomization_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    for r in rows:
        print("%-22s box %8.1f ms  verify %7.1f ms  transcript %9d B  unlinkable %d  servers %d" %
              (r["method"], r["box_ms"], r["verify_ms"], r["bytes"],
               r["hides_link"], r["servers"]))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
