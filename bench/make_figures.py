import csv
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
})


def read_csv(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    eps = os.path.join(FIGS, name + ".eps")
    fig.savefig(eps, format="eps", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    subprocess.run(["epstopdf", eps, "--outfile=" + os.path.join(FIGS, name + ".pdf")],
                   check=False)
    if not os.path.exists(os.path.join(FIGS, name + ".pdf")):
        subprocess.run(["ps2pdf", "-dEPSCrop", eps, os.path.join(FIGS, name + ".pdf")],
                       check=False)
    print("wrote", name)


def fig_randomization():
    rows = read_csv("randomization.csv")
    labels = [r["method"].replace("ReRand+", "RR+") for r in rows]
    kb = [max(float(r["bytes"]) / 1024.0, 0.05) for r in rows]
    box = [float(r["box_ms"]) for r in rows]
    unlink = [int(r["hides_link"]) for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    colors = ["#c8c8c8" if u == 0 else "#4a6fa5" for u in unlink]
    colors[-1] = "#b03a2e"
    axes[0].bar(x, kb, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("added transcript per ballot (KiB)")
    axes[1].bar(x, box, color=colors, edgecolor="black", linewidth=0.4)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("processing time per ballot (ms)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="#c8c8c8", ec="black", lw=0.4),
               plt.Rectangle((0, 0), 1, 1, fc="#4a6fa5", ec="black", lw=0.4),
               plt.Rectangle((0, 0), 1, 1, fc="#b03a2e", ec="black", lw=0.4)]
    axes[0].legend(handles, ["link is public", "link hidden", "ABL (this work)"],
                   loc="upper right")
    save(fig, "fig_randomization")


def fig_tally():
    rows = read_csv("tally.csv")
    v = [int(r["voters"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    axes[0].loglog(v, [float(r["jcj_seconds"]) for r in rows], "o--", color="#7f7f7f", label="JCJ", ms=3.5)
    axes[0].loglog(v, [float(r["chide_seconds"]) for r in rows], "s--", color="#4a6fa5", label="CHide", ms=3.5)
    axes[0].loglog(v, [float(r["simurgh_seconds"]) for r in rows], "^-", color="#b03a2e", label="Simurgh", ms=3.5)
    axes[0].set_xlabel("registered voters")
    axes[0].set_ylabel("tally time (s)")
    axes[0].legend()
    axes[1].loglog(v, [float(r["jcj_bytes"]) / 1e9 for r in rows], "o--", color="#7f7f7f", label="JCJ", ms=3.5)
    axes[1].loglog(v, [float(r["chide_bytes"]) / 1e9 for r in rows], "s--", color="#4a6fa5", label="CHide", ms=3.5)
    axes[1].loglog(v, [float(r["simurgh_bytes"]) / 1e9 for r in rows], "^-", color="#b03a2e", label="Simurgh", ms=3.5)
    axes[1].set_xlabel("registered voters")
    axes[1].set_ylabel("tally transcript (GB)")
    axes[1].legend()
    save(fig, "fig_tally")


def fig_ballot():
    rows = read_csv("ballot.csv")
    nc = [int(r["options"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.plot(nc, [float(r["ballot_bytes"]) / 1024 for r in rows], "o-", color="#b03a2e", ms=3.5, label="ballot")
    ax.plot(nc, [float(r["proof_bytes"]) / 1024 for r in rows], "s--", color="#4a6fa5", ms=3.5, label="validity proof")
    ax.set_xlabel("voting options $n_C$")
    ax.set_ylabel("size (KiB)")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(nc, [float(r["verify_ms"]) for r in rows], "^:", color="#2e7d32", ms=3.5, label="verify")
    ax2.set_ylabel("verification (ms)")
    ax2.grid(False)
    ax2.legend(loc="center right")
    save(fig, "fig_ballot")


def fig_election():
    if not os.path.exists(os.path.join(RESULTS, "election.csv")):
        return
    rows = read_csv("election.csv")
    v = [int(r["voters"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for key, style, col, lab in (("setup_s", "o-", "#7f7f7f", "setup"),
                                 ("register_s", "s-", "#4a6fa5", "registration"),
                                 ("cast_s", "^-", "#b03a2e", "casting"),
                                 ("tally_s", "d-", "#2e7d32", "tally"),
                                 ("verify_s", "v-", "#8e44ad", "verification")):
        ax.plot(v, [float(r[key]) for r in rows], style, color=col, ms=3.5, label=lab)
    ax.set_xlabel("registered voters")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(ncol=2)
    save(fig, "fig_election")


if __name__ == "__main__":
    fig_randomization()
    fig_tally()
    fig_ballot()
    fig_election()
