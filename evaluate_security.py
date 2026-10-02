"""
evaluate_security.py - Benchmarking of the jamming defense.

  1. Jamming power vs spectral efficiency (P_j sweep) for: No-RIS, random phase, unprotected AI-RIS,
     proposed middleware, and the null-steering ZF reference.
  2. Latency / overhead of the middleware per channel realization vs the RIC control-loop budget.

Outputs (in --outdir): fig_pj_sweep.{png,pdf}, fig_latency.{png,pdf}, pj_sweep.csv, latency.csv, summary.md
"""
import argparse
import csv
import os
import platform
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from channel_model import WirelessChannel

# baselines_2.py veya baselines.py uyumlu dinamik import
try:
    from baselines import BaselineSolvers, quantize_phase
except ImportError:
    from baselines_2 import BaselineSolvers, quantize_phase

try:
    import torch
    from security_middleware import SecurityMiddleware

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
M_ANT = 4
SCENARIOS = {
    "A": {"label": "A: jammer close to user, N=32", "N": 32, "jammer_pos": (55.0, -15.0, 1.5)},
    "B": {"label": "B: distant jammer, N=64", "N": 64, "jammer_pos": (30.0, -60.0, 1.5)},
}
BUDGET_MS = 10.0  # target requested for decision inference
NEAR_RT_MAX_MS = 1000.0  # upper end of the O-RAN Near-RT RIC control loop (10 ms - 1 s)

# Optimizer configurations traded off in the latency study
OPT_CONFIGS = {
    "full (8 restarts, 200+100 steps, polish)": {},
    "balanced (4 restarts, 60+30 steps)": dict(num_restarts=4, cont_steps=60, qat_steps=30, polish_sweeps=1),
    "fast (2 restarts, 20+10 steps)": dict(num_restarts=2, cont_steps=20, qat_steps=10, polish_sweeps=0),
    "minimal (1 restart, 10+5 steps)": dict(num_restarts=1, cont_steps=10, qat_steps=5, polish_sweeps=0),
}

M_REF, M_NORIS, M_RAND = "Unprotected, jammer off", "No-RIS", "Random phase RIS"
M_UNPROT, M_PROP, M_ZF = "Unprotected AI-RIS", "Proposed middleware", "Null-steering ZF"
STYLE = {  # Okabe-Ito colour-blind-safe palette
    M_NORIS: dict(color="#999999", marker="v", ls="-"),
    M_RAND: dict(color="#E69F00", marker="s", ls="-"),
    M_UNPROT: dict(color="#D55E00", marker="o", ls="-"),
    M_PROP: dict(color="#0072B2", marker="D", ls="-", lw=2.2),
    M_ZF: dict(color="#009E73", marker="^", ls="--"),
    M_REF: dict(color="black", marker=None, ls=":"),
}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.markersize": 4.5,
})


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def noisy_estimate(x, nmse, rng):
    """Channel estimate with the given NMSE (same model as environment.py)."""
    if nmse <= 0.0:
        return x
    err = (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)) / np.sqrt(2.0)
    return x + np.sqrt(nmse * np.mean(np.abs(x) ** 2)) * err


def make_estimates(ch, nmse, rng):
    """Channels the middleware sees: exact legitimate channels, noisy interference channels."""
    h_d, G, h_r, h_ju, G_j = ch
    return {
        "h_d": h_d,
        "G": G,
        "h_r": h_r,
        "h_ju": noisy_estimate(h_ju, nmse, rng),
        "G_j": noisy_estimate(G_j, nmse, rng)
    }


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def lat_stats(ms):
    a = np.asarray(ms, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save_fig(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)


# ----------------------------------------------------------------------
# 1) Jamming power sweep
# ----------------------------------------------------------------------
def run_sweep(key, pj_list, trials, bits, nmse, seed, opt_kwargs, include_proposed):
    sc = SCENARIOS[key]
    N = sc["N"]
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sim = WirelessChannel(num_antennas=M_ANT, num_elements=N, jammer_pos=sc["jammer_pos"])

    # Same channels, random phases and estimates are reused at every P_j (paired comparison).
    base = BaselineSolvers(M_ANT, N, phase_bits=bits)
    trials_data = []
    for _ in range(trials):
        ch0, ch1 = sim.get_channel_realization(), sim.get_channel_realization()
        ao = base.unprotected_ris(*ch1)[1]
        m_zf, zf_phases = base.analytical_null_steering(*ch1)
        trials_data.append(dict(
            ch0=ch0, ch1=ch1, est0=make_estimates(ch0, nmse, rng), est1=make_estimates(ch1, nmse, rng),
            ao=ao, rand=quantize_phase(rng.uniform(0.0, 2.0 * np.pi, N), bits),
            zf=zf_phases, zf_feasible=m_zf["null_feasible"]))

    rows, detect = [], {}
    for pj in pj_list:
        solvers = BaselineSolvers(M_ANT, N, jammer_power_dbm=pj, phase_bits=bits)
        n_det = 0
        for t, td in enumerate(trials_data):
            ch1 = td["ch1"]
            res = {
                M_NORIS: solvers.evaluate(*ch1, phases=None),
                M_RAND: solvers.evaluate(*ch1, phases=td["rand"]),
                M_UNPROT: solvers.evaluate(*ch1, phases=td["ao"]),
                M_ZF: solvers.evaluate(*ch1, phases=td["zf"]),
            }
            res[M_REF] = {"rate": res[M_UNPROT]["rate_no_jam"], "sinr_db": res[M_UNPROT]["snr_db"], "inr_db": np.nan}

            if include_proposed:
                mw = SecurityMiddleware(N, solvers.P_t, solvers.sigma2, bits=bits, seed=seed + t, **opt_kwargs)
                ph0 = mw.reconfigure(td["est0"])  # slot 0: normal mode
                m0 = solvers.evaluate(*td["ch0"], phases=ph0)
                attack = mw.monitor(td["est0"], measured_sinr_db=m0["sinr_db"])
                n_det += bool(attack)
                ph1 = mw.reconfigure(td["est1"])  # slot 1: mitigation
                res[M_PROP] = solvers.evaluate(*ch1, phases=ph1)

            for method, m in res.items():
                rows.append(dict(scenario=key, pj_dbm=pj, trial=t, method=method, rate=m["rate"],
                                 sinr_db=m["sinr_db"], inr_db=m["inr_db"]))
        detect[pj] = (n_det / trials) if include_proposed else np.nan
    feas = float(np.mean([td["zf_feasible"] for td in trials_data]))
    return rows, detect, feas


def aggregate(rows):
    agg = {}
    for r in rows:
        agg.setdefault((r["scenario"], r["method"], r["pj_dbm"]), []).append(r["rate"])
    return {k: mean_ci(v) for k, v in agg.items()}


def plot_sweep(agg, detect, feas, pj_list, keys, include_proposed, outdir):
    methods = [M_NORIS, M_RAND, M_UNPROT] + ([M_PROP] if include_proposed else []) + [M_ZF]
    n = len(keys)
    fig = plt.figure(figsize=(3.8 * n, 4.5 if include_proposed else 3.5))
    gs = fig.add_gridspec(2 if include_proposed else 1, n, height_ratios=[3, 1] if include_proposed else [1],
                          hspace=0.18, wspace=0.28)
    handles, labels = None, None

    for j, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, j])
        for m in methods:
            mu = np.array([agg[(key, m, p)][0] for p in pj_list])
            ci = np.array([agg[(key, m, p)][1] for p in pj_list])
            st = dict(STYLE[m])
            ax.plot(pj_list, mu, label=m, lw=st.pop("lw", 1.4), **st)
            ax.fill_between(pj_list, mu - ci, mu + ci, color=STYLE[m]["color"], alpha=0.15, lw=0)

        ref = np.mean([agg[(key, M_REF, p)][0] for p in pj_list])
        ax.axhline(ref, **{k: v for k, v in STYLE[M_REF].items() if k in ("color", "ls")}, lw=1, label=M_REF)
        ax.set_title(f"{SCENARIOS[key]['label']}\n(exact null feasible in {100 * feas[key]:.0f}% of trials)")
        ax.set_ylabel("Spectral efficiency (bps/Hz)")
        ax.set_ylim(bottom=0)

        if include_proposed:
            ax.tick_params(labelbottom=False)
            ax2 = fig.add_subplot(gs[1, j], sharex=ax)
            ax2.plot(pj_list, [100.0 * detect[(key, p)] for p in pj_list], color=STYLE[M_PROP]["color"], marker="D")
            ax2.set_ylim(-5, 105)
            ax2.set_ylabel("Detected (%)")
            ax2.set_xlabel("Jammer transmit power $P_j$ (dBm)")
        else:
            ax.set_xlabel("Jammer transmit power $P_j$ (dBm)")

        if j == 0:
            handles, labels = ax.get_legend_handles_labels()

    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Spectral efficiency under jamming (mean $\\pm$ 95% CI)", y=1.02)
    save_fig(fig, outdir, "fig_pj_sweep")


# ----------------------------------------------------------------------
# 2) Latency & overhead
# ----------------------------------------------------------------------
def run_latency(key, runs, warmup, bits, nmse, seed, pj_dbm=30.0):
    sc = SCENARIOS[key]
    N = sc["N"]
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sim = WirelessChannel(num_antennas=M_ANT, num_elements=N, jammer_pos=sc["jammer_pos"])
    solvers = BaselineSolvers(M_ANT, N, jammer_power_dbm=pj_dbm, phase_bits=bits)
    chans = [sim.get_channel_realization() for _ in range(runs + warmup)]
    ests = [make_estimates(c, nmse, rng) for c in chans]

    records = []

    # Geleneksel baselines (kıyaslama referansları)
    for name, fn, k in (("Unprotected AO (numpy)", lambda c: solvers._ao_phases(*c[:3]), runs),
                        ("Null-steering ZF (numpy)", lambda c: solvers.analytical_null_steering(*c), min(runs, 30))):
        ts = []
        for c in chans[warmup:warmup + k]:
            t0 = time.perf_counter()
            fn(c)
            ts.append((time.perf_counter() - t0) * 1e3)
        records.append((name, "-", ts, np.nan))

    if HAVE_TORCH:
        for cfg_name, kw in OPT_CONFIGS.items():
            mw = SecurityMiddleware(N, solvers.P_t, solvers.sigma2, bits=bits, seed=seed, probe_interval=0, **kw)
            mw.detector.under_attack = True
            mw.p_j_hat_w = solvers.P_j
            t_rec, t_mon, t_e2e, rates = [], [], [], []
            for i, (c, e) in enumerate(zip(chans, ests)):
                t0 = time.perf_counter()
                ph = mw.reconfigure(e)
                t1 = time.perf_counter()
                m = solvers.evaluate(*c, phases=ph)
                t2 = time.perf_counter()
                mw.monitor(e, measured_sinr_db=m["sinr_db"])
                t3 = time.perf_counter()
                if i >= warmup:
                    t_rec.append((t1 - t0) * 1e3)
                    t_mon.append((t3 - t2) * 1e3)
                    t_e2e.append((t1 - t0 + t3 - t2) * 1e3)
                    rates.append(m["rate"])
            records.append(("Detector (monitor)", cfg_name, t_mon, np.nan))
            records.append(("Phase optimization (reconfigure)", cfg_name, t_rec, float(np.mean(rates))))
            records.append(("End-to-end decision", cfg_name, t_e2e, float(np.mean(rates))))
    return records


def plot_latency(records, outdir):
    sel = [r for r in records if r[0] in ("End-to-end decision", "Detector (monitor)") or r[0].endswith("(numpy)")]
    sel = [r for r in sel if r[0] != "Detector (monitor)" or r[1] == next(iter(OPT_CONFIGS))]
    labels = [f"{c} - {n}" if c != "-" else n for n, c, _, _ in sel]

    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.55 * len(sel) + 1.2)))
    bp = ax.boxplot([r[2] for r in sel], vert=False, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=1.5))
    for patch in bp["boxes"]:
        patch.set(facecolor="#56B4E9", alpha=0.7)
    ax.set_yticks(range(1, len(sel) + 1))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(BUDGET_MS, color="#D55E00", ls="--", lw=1.2, label=f"{BUDGET_MS:.0f} ms target")
    ax.axvline(NEAR_RT_MAX_MS, color="#CC79A7", ls="--", lw=1.2, label="1 s (Near-RT RIC upper bound)")
    ax.set_xlabel("Latency per channel realization (ms, log scale; box = IQR, whiskers = 5-95th pct)")
    ax.legend(loc="upper left")
    save_fig(fig, outdir, "fig_latency")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30, help="channel realizations per P_j point")
    ap.add_argument("--pj-dbm", type=float, nargs="+", default=[10, 15, 20, 25, 30, 35, 40])
    ap.add_argument("--scenarios", nargs="+", default=["A", "B"], choices=list(SCENARIOS))
    ap.add_argument("--bits", type=int, default=2, choices=[1, 2], help="RIS phase resolution")
    ap.add_argument("--nmse", type=float, default=0.05, help="interference-CSI NMSE seen by the middleware")
    ap.add_argument("--latency-runs", type=int, default=200)
    ap.add_argument("--latency-scenario", default="B", choices=list(SCENARIOS))
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 = lowest latency for tiny tensors)")
    ap.add_argument("--sweep-config", default=next(iter(OPT_CONFIGS)), choices=list(OPT_CONFIGS),
                    help="optimizer configuration used in the P_j sweep")
    ap.add_argument("--skip-proposed", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args(argv)
    if a.quick:
        a.trials, a.latency_runs, a.pj_dbm = 5, 20, [10, 25, 40]

    include_proposed = HAVE_TORCH and not a.skip_proposed
    if not HAVE_TORCH and not a.skip_proposed:
        print("PyTorch not found -> proposed middleware and its latency are skipped.")
    if HAVE_TORCH:
        torch.set_num_threads(a.threads)
    os.makedirs(a.outdir, exist_ok=True)

    env_info = f"Python {platform.python_version()} | numpy {np.__version__} | {platform.processor() or platform.machine()}"
    if HAVE_TORCH:
        env_info += f" | torch {torch.__version__} ({torch.get_num_threads()} thread(s), CPU)"
    print(env_info)

    # ---- sweep ----
    pj_list = list(a.pj_dbm)
    all_rows, detect, feas = [], {}, {}
    for key in a.scenarios:
        print(f"\nP_j sweep, scenario {SCENARIOS[key]['label']} ...")
        rows, det, f = run_sweep(key, pj_list, a.trials, a.bits, a.nmse, a.seed,
                                 OPT_CONFIGS[a.sweep_config], include_proposed)
        all_rows += rows
        detect.update({(key, p): v for p, v in det.items()})
        feas[key] = f

    agg = aggregate(all_rows)
    plot_sweep(agg, detect, feas, pj_list, a.scenarios, include_proposed, a.outdir)

    with open(os.path.join(a.outdir, "pj_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    md = [f"# Jamming defense evaluation\n\n{env_info}\n\n{a.trials} trials/point, {a.bits}-bit RIS, "
          f"interference-CSI NMSE {a.nmse}, M={M_ANT}.\n"]
    methods = [M_NORIS, M_RAND, M_UNPROT] + ([M_PROP] if include_proposed else []) + [M_ZF]
    for key in a.scenarios:
        ref = np.mean([agg[(key, M_REF, p)][0] for p in pj_list])
        headers = ["P_j (dBm)"] + methods + (["Proposed / jammer-off", "Attack detected"] if include_proposed else [])
        trows = []
        for p in pj_list:
            row = [f"{p:g}"] + [f"{agg[(key, m, p)][0]:.2f} ± {agg[(key, m, p)][1]:.2f}" for m in methods]
            if include_proposed:
                row += [f"{100 * agg[(key, M_PROP, p)][0] / ref:.0f}%", f"{100 * detect[(key, p)]:.0f}%"]
            trows.append(row)
        md.append(f"\n## Scenario {SCENARIOS[key]['label']}\n\nJammer-off reference: {ref:.2f} bps/Hz. "
                  f"Exact null feasible in {100 * feas[key]:.0f}% of trials (ZF is a fallback otherwise).\n\n"
                  + md_table(headers, trows) + "\n")

    # ---- latency ----
    print(f"\nLatency benchmark, scenario {SCENARIOS[a.latency_scenario]['label']} ...")
    records = run_latency(a.latency_scenario, a.latency_runs, warmup=5, bits=a.bits, nmse=a.nmse, seed=a.seed)
    plot_latency(records, a.outdir)
    lrows, csv_rows = [], []
    for comp, cfg, ts, rate in records:
        s = lat_stats(ts)
        verdict = ("PASS" if s["p99"] < BUDGET_MS else "FAIL") + " / " + (
            "PASS" if s["p99"] < NEAR_RT_MAX_MS else "FAIL")
        lrows.append([comp, cfg, f"{s['mean']:.2f}", f"{s['median']:.2f}", f"{s['p95']:.2f}", f"{s['p99']:.2f}",
                      f"{s['max']:.2f}", verdict, "-" if np.isnan(rate) else f"{rate:.2f}"])
        csv_rows.append(dict(component=comp, config=cfg, runs=len(ts), **{k: round(v, 4) for k, v in s.items()},
                             mean_rate=rate))

    with open(os.path.join(a.outdir, "latency.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    lat_headers = ["Component", "Config", "mean (ms)", "median", "p95", "p99", "max", "p99 <10 ms / <1 s",
                   "rate (bps/Hz)"]
    md.append(f"\n## Latency (scenario {a.latency_scenario}, {a.latency_runs} realizations, P_j = 30 dBm)\n\n"
              f"Verdicts use the p99 latency. Rate is the mean spectral efficiency under attack for that optimizer "
              f"configuration.\n\n" + md_table(lat_headers, lrows) + "\n")
    if not HAVE_TORCH:
        md.append("\n*PyTorch not installed: only the numpy reference solvers were timed.*\n")

    with open(os.path.join(a.outdir, "summary.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"\nSaved figures, CSVs and summary.md to {a.outdir}/")


if __name__ == "__main__":
    main()