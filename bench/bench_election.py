import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simurgh.scheme import Simurgh

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run(n_voters, n_c, seed):
    S = Simurgh(n_c=n_c, n_voters=n_voters, seed=seed)
    t0 = time.perf_counter()
    S.setup()
    t_setup = time.perf_counter() - t0
    t0 = time.perf_counter()
    creds = [S.register(i) for i in range(n_voters)]
    t_reg = time.perf_counter() - t0
    for i, c in enumerate(creds):
        assert S.open_real(c) == S.real_slot[i]
        assert S.fake_cred(c) == S.decoy_slot[i]
    rng = np.random.default_rng(seed + 1)
    choices = [int(rng.integers(0, n_c)) for _ in range(n_voters)]
    t0 = time.perf_counter()
    for i in range(n_voters):
        b = S.vote(S.real_slot[i], choices[i], seq=1)
        assert S.cast(b)
    t_cast = time.perf_counter() - t0
    coerced = S.vote(S.decoy_slot[0], (choices[0] + 1) % n_c, seq=1)
    assert S.cast(coerced)
    t0 = time.perf_counter()
    result, transcript = S.tally()
    t_tally = time.perf_counter() - t0
    t0 = time.perf_counter()
    ok = S.verify_tally(result, transcript)
    t_ver = time.perf_counter() - t0
    expected = [sum(1 for c in choices if c == j) for j in range(n_c)]
    assert result == expected, (result, expected)
    assert ok
    return {
        "voters": n_voters,
        "options": n_c,
        "setup_s": t_setup,
        "register_s": t_reg,
        "cast_s": t_cast,
        "cast_per_ballot_s": t_cast / n_voters,
        "tally_s": t_tally,
        "verify_s": t_ver,
        "transcript_bytes": S.transcript_bytes(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voters", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--options", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rows = []
    for n in args.voters:
        row = run(n, args.options, args.seed)
        rows.append(row)
        print(row)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "election.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
