import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simurgh.fahe import HomPKE
from simurgh.params import find_prime
from simurgh.proofs import BallotProof, TAU, proof_widths, hash_state, proof_widths
from simurgh.ring import Rq, ntt_primes

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--options", type=int, nargs="+", default=[2, 4, 8, 16])
    args = ap.parse_args()

    rng = np.random.default_rng(11)
    R = Rq(4096, ntt_primes(4096, 3, 28))
    E = HomPKE(R, find_prime(1 << 17, 8, 3), 3.2)
    pk, sk = E.keygen(rng)
    ct_bytes = 2 * R.n * R.logq // 8
    rows = []
    for n_c in args.options:
        s1, s2, m = proof_widths(R, n_c)
        proof = BallotProof(E, n_c, s1, s2, m)
        vt, bt, rt, rounds = [], [], [], []
        for _ in range(args.reps):
            k = int(rng.integers(0, n_c))
            msg = np.zeros(R.n, dtype=np.int64)
            msg[k] = 1
            ct, wit = E.encrypt(pk, rng, msg)
            ctx = hash_state(b"bench", 0, 1)
            tries = 0
            t_voter = 0.0
            t_box = 0.0
            while True:
                tries += 1
                t0 = time.perf_counter()
                alphas, seeds, zs, raws = proof.commit(pk, rng, ct, k)
                t_voter += time.perf_counter() - t0
                t0 = time.perf_counter()
                ct2, alphas2, aux = proof.launder_commit(pk, rng, ct, alphas)
                t_box += time.perf_counter() - t0
                t0 = time.perf_counter()
                r = proof.respond(pk, rng, k, wit, None, seeds, zs, raws, ctx, ct2, alphas2)
                t_voter += time.perf_counter() - t0
                if r is None:
                    continue
                seeds2, zs2, raws2 = r
                t0 = time.perf_counter()
                fin = proof.finalize(rng, seeds2, zs2, raws2, aux)
                t_box += time.perf_counter() - t0
                if fin is None:
                    continue
                zs3, _ = fin
                break
            t0 = time.perf_counter()
            assert proof.verify(pk, ct2, ctx, alphas2, seeds2, zs3)
            rt.append(time.perf_counter() - t0)
            vt.append(t_voter)
            bt.append(t_box)
            rounds.append(tries)
        rows.append({
            "options": n_c,
            "voter_ms": 1000 * float(np.mean(vt)),
            "box_ms": 1000 * float(np.mean(bt)),
            "verify_ms": 1000 * float(np.mean(rt)),
            "ballot_bytes": ct_bytes + proof.proof_bytes() + 3309,
            "proof_bytes": proof.proof_bytes(),
            "rounds": float(np.mean(rounds)),
        })
        print(rows[-1])

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "ballot.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULTS, "ballot_meta.json"), "w") as fh:
        json.dump({"degree": R.n, "logq": R.logq, "ciphertext_bytes": ct_bytes,
                   "signature_bytes": 3309}, fh, indent=2)


if __name__ == "__main__":
    main()
