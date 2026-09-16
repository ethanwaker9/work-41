# Post-Quantum Coercion Resistant Voting System From Anamorphic Encryption

This repository contains our research on **Simurgh**, a post-quantum internet voting scheme that is
receipt-free, coercion resistant and strongly verifiable, together with the benchmarks. The scheme is built from four primitives only as fully asymmetric anamorphic homomorphic encryption (FAHE) over module lattices, a lattice distributed key generation with
verifiable short secret sharing, ML-DSA and SHAKE256.

## Files and Content

```
simurgh/ring.py        negacyclic ring arithmetic with a residue number system NTT
simurgh/params.py      parameter sets and a primal-uSVP core-SVP security estimator
simurgh/fahe.py        FAHE: Regev normal mode and the anamorphic dual-Regev triplet
simurgh/dkg.py         verifiable short secret sharing, threshold decryption, flooding
simurgh/proofs.py      the Sigma-OR ballot proof and anamorphic ballot laundering
simurgh/weights.py     batched proof that a trustee applied its committed weight shares
simurgh/baselines.py   the five post-quantum ballot randomisation methods compared
simurgh/classical.py   ElGamal, JCJ and CHide cleansing cost models
simurgh/scheme.py      Setup, Register, Vote, Valid, Tally, Verify
bench/                 benchmark drivers, all of which write CSV files to results/
tests/                 unit tests
```

## Running Experiments

```
python3 -m pip install -r requirements.txt
```

```
python3 run_all.py
```
This runs the unit tests, all four benchmarks and the result generation. Results are
written as CSV files to `results/` and as EPS and PDF files to `figures/`. Individual
benchmarks can be run on their own, for example
```
python3 bench/bench_randomization.py --reps 10
python3 bench/bench_tally.py --reps 8
python3 bench/bench_ballot.py --options 2 4 8 16
python3 bench/bench_election.py --voters 8 16 32 64
```
Every benchmark accepts `--reps` to trade running time against variance. The tally
benchmark measures the cost of one plaintext equality test, one conditional gate and one
weighted slot aggregation, and then projects the three cleansing procedures to the
electorate sizes reported in the paper.

## Minimal Example

```python
from simurgh.scheme import Simurgh

S = Simurgh(n_c=3, n_voters=4, seed=17)
S.setup()
creds = [S.register(i) for i in range(4)]

# the capsule delivers the real slot covertly and the decoy slot in the clear
assert S.open_real(creds[0]) == S.real_slot[0]
assert S.fake_cred(creds[0]) == S.decoy_slot[0]

for i, v in enumerate([0, 1, 1, 2]):
    S.cast(S.vote(S.real_slot[i], v, seq=1))

# a coercer voting with the surrendered credential produces a publicly valid ballot
S.cast(S.vote(S.decoy_slot[0], 2, seq=1))

result, transcript = S.tally()
assert result == [1, 2, 1] and S.verify_tally(result, transcript)
```

