import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ([sys.executable, "-m", "pytest", "tests", "-q"], "unit tests"),
    ([sys.executable, "bench/bench_randomization.py", "--reps", "5"], "ballot randomisation"),
    ([sys.executable, "bench/bench_ballot.py", "--reps", "5"], "ballot cost"),
    ([sys.executable, "bench/bench_tally.py", "--reps", "5"], "tally scaling"),
    ([sys.executable, "bench/bench_election.py", "--voters", "8", "16", "32", "64"], "full election"),
    ([sys.executable, "bench/make_figures.py"], "figures"),
]

if __name__ == "__main__":
    for cmd, label in STEPS:
        print("\n=== %s ===" % label)
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("\nall artefacts written to results/ and figures/")
