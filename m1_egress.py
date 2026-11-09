#!/usr/bin/env python3
"""
M1 - congested baseline egress (Bailey Hall).  Builds on m0_egress.py (keep both files together).

Everyone leaves their seat at t = 0, reaches the aisle after t_row (Eq. 3), then follows the M0
fastest route.  Every node is a first-in-first-out server with capacity c_n = w_n * phi:

    alpha_(1) -> delta_(1) = alpha_(1)
    delta_(j) = max( alpha_(j), delta_(j-1) + 1/c_n )          departure from node n
    alpha_{i,q} = delta_{i,n} + w_nq                           arrival at next node q
    T_i = delta_{i, x_i}      T = max_i T_i

Cases (no behaviour, so concert and lecture emergencies are the same in M1):
    ordinary   only --ordinary-exits open, speeds x --speed-scale
    emergency  all exits open

Outputs (in --out), for each case:
    m1_<case>_seat_times.csv   T_m0, T_m1, delay per seat
    m1_<case>_heatmap.png      T_i under congestion
    m1_<case>_delay.png        T_m1 - T_m0  (where the crowd hurts)
    m1_<case>_nodes.csv        served, total wait, max wait per node (largest wait = bottleneck)
    m1_summary.csv             T, T90, mean per case and level

Usage:  python3 m1_egress.py auditorium_graph_coding.xlsx --out m1_output [--verify]
   or:  python3 m1_egress.py --seats seats.csv --nodes nodes.csv --edges edges.csv --out m1_output
        (csv columns must match the M0 workbook: seat_id, level, row_label, seat_number, node_id,
         seats_to_aisle / node_id, node_type / edge_id, node_1_id, node_2_id, edge_type, walking_distance_*)
        --widths widths.csv   (columns node_id,width_m) to supply measured widths
        nodes sheet may also hold width_m / width_inches / width_ft columns.

!! Widths by node type and the specific flows below are PLACEHOLDERS until measured / cited.
"""
import argparse
import heapq
import math
import os
import random
import sys

import numpy as np
import pandas as pd

import m0_egress as m0
from m0_egress import INCH

M1 = dict(
    phi_door=1.3,      # persons / s / m  (doors, level aisles)   PLACEHOLDER
    phi_stairs=0.9,    # persons / s / m  (stairs, stepped aisles) PLACEHOLDER
    width_m={"door": 1.2, "stairs": 1.5, "aisle_stairs": 1.2,
             "aisle_concrete": 1.0, "aisle_wood": 1.0, "default": 1.2},   # PLACEHOLDER
    ordinary_speed_scale=0.7,
)


# ----------------------------------------------------------------------------
# capacities  c_n = w_n * phi
# ----------------------------------------------------------------------------
def node_widths(nodes, widths_csv=None):
    w = {}
    for r in nodes.to_dict("records"):
        v = np.nan
        for col, k in (("width_m", 1.0), ("width_inches", INCH), ("width_ft", 0.3048)):
            x = pd.to_numeric(r.get(col), errors="coerce")
            if pd.notna(x):
                v = float(x) * k
                break
        w[r["node_id"]] = v
    if widths_csv:
        df = pd.read_csv(widths_csv)
        for n, x in zip(df["node_id"], pd.to_numeric(df["width_m"], errors="coerce")):
            if n not in w:
                sys.exit(f"--widths: unknown node {n}")
            w[n] = x
    return w


def capacities(nodes, all_exits, widths):
    """Exit nodes are uncapacitated unless a width is given; others fall back to placeholders."""
    cap, defaulted = {}, []
    for r in nodes.to_dict("records"):
        n, t = r["node_id"], str(r.get("node_type"))
        w = widths.get(n, np.nan)
        if n in all_exits and not np.isfinite(w):
            cap[n] = math.inf
            continue
        if not np.isfinite(w):
            w = M1["width_m"].get(t, M1["width_m"]["default"])
            defaulted.append(n)
        cap[n] = w * (M1["phi_stairs"] if "stairs" in t else M1["phi_door"])
    return cap, defaulted


# ----------------------------------------------------------------------------
# M0 routes: f(x) plus the minimizing next node
# ----------------------------------------------------------------------------
def potential_with_next(adj, order, exits_open):
    f = {n: math.inf for n in order}
    nxt = {}
    for x in exits_open:
        f[x] = 0.0
    for p in reversed(order):
        for q, w in adj.get(p, []):
            c = w + f[q]
            if c < f[p]:
                f[p], nxt[p] = c, (q, w)
    return f, nxt


# ----------------------------------------------------------------------------
# event-driven queueing simulation
# ----------------------------------------------------------------------------
def simulate(start_nodes, t_rows, nxt, cap, exits_open, seed=0):
    rng = random.Random(seed)
    inv = {n: (1.0 / c if np.isfinite(c) and c > 0 else 0.0) for n, c in cap.items()}
    last = {}
    heap = [(float(t), rng.random(), i, n) for i, (n, t) in enumerate(zip(start_nodes, t_rows))]
    heapq.heapify(heap)
    exits = set(exits_open)
    T = np.full(len(start_nodes), np.nan)
    stats = {n: dict(served=0, total_wait=0.0, max_wait=0.0) for n in cap}
    while heap:
        t, _, i, n = heapq.heappop(heap)                 # earliest event; ties broken at random
        d = max(t, last.get(n, -math.inf) + inv[n])      # departure from node n
        last[n] = d
        s = stats[n]
        s["served"] += 1
        s["total_wait"] += d - t
        s["max_wait"] = max(s["max_wait"], d - t)
        if n in exits:
            T[i] = d
            continue
        q, w = nxt[n]
        heapq.heappush(heap, (d + w, rng.random(), i, q))   # arrival at next node
    return T, stats


# ----------------------------------------------------------------------------
# plotting (same layout as the M0 heat map)
# ----------------------------------------------------------------------------
def plot_map(res, col, label, title, path, real_coords, vmin=None, vmax=None, cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [l for l in ("orchestra", "lodge", "balcony") if l in set(res.level)]
    ok = res.dropna(subset=[col])
    vmin = ok[col].min() if vmin is None else vmin
    vmax = ok[col].max() if vmax is None else vmax
    ratios = [max(res[res.level == l]["y"].nunique(), 3) for l in levels]
    fig, axes = plt.subplots(len(levels), 1, figsize=(12, 2.2 * sum(ratios) / 4 + 1.5),
                             gridspec_kw={"height_ratios": ratios}, squeeze=False)
    sc = None
    for ax, level in zip(axes[:, 0], levels):
        g = res[res.level == level]
        width = g.groupby("y")["x"].count().max()
        size = float(np.clip((12 * 72 / (width + 14)) ** 2 * 0.3, 4, 90))
        sc = ax.scatter(g["x"], g["y"], c=g[col], s=size, marker="s", cmap=cmap, vmin=vmin, vmax=vmax)
        bad = g[g[col].isna()]
        if len(bad):
            ax.scatter(bad["x"], bad["y"], s=size, marker="x", c="red")
        gk = g.dropna(subset=[col])
        if len(gk):
            s = gk.loc[gk[col].idxmax()]
            ax.scatter([s["x"]], [s["y"]], s=size * 4, facecolors="none", edgecolors="red", linewidths=1.5)
            ax.annotate(f"{s['seat_id']}  {s[col]:.0f}s", (s["x"], s["y"]), xytext=(8, 8),
                        textcoords="offset points", color="red", fontsize=8)
        if not real_coords:
            rows = sorted(g.row_label.unique())
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows, fontsize=7)
            ax.set_xticks([])
        ax.set_title(level, loc="left", fontsize=10)
        ax.margins(0.02)
    fig.colorbar(sc, ax=axes[:, 0].tolist(), label=label, shrink=0.8)
    fig.suptitle(title + ("" if real_coords else "  (schematic layout, not to scale)"), fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# one case
# ----------------------------------------------------------------------------
def run_case(name, exits_open, speed_scale, seats, nodes, edges, all_nodes, cap, a):
    P = dict(m0.PARAMS)
    P["speed_mps"] = {k: v * speed_scale for k, v in m0.PARAMS["speed_mps"].items()}
    P["v_seat_mps"] = m0.PARAMS["v_seat_mps"] * speed_scale
    adj, _ = m0.build_graph(nodes, edges, P, a.skip_missing)
    order = m0.topological_order(adj, all_nodes)
    f, exit_of = m0.egress_potential(adj, order, exits_open)
    _, nxt = potential_with_next(adj, order, exits_open)
    res = m0.seat_times(seats, f, exit_of, P).rename(columns={"T_s": "T_m0_s"})
    ok = res["T_m0_s"].notna().values
    idx = np.flatnonzero(ok)
    starts = res["node_id"].values[idx]
    rows = res["t_row_s"].values[idx]

    T, stats = simulate(starts, rows, nxt, cap, exits_open, a.seed)
    res["T_m1_s"] = np.nan
    res.loc[res.index[idx], "T_m1_s"] = T
    res["delay_s"] = res["T_m1_s"] - res["T_m0_s"]

    if a.verify:
        inf_cap = {n: math.inf for n in cap}
        T_inf, _ = simulate(starts, rows, nxt, inf_cap, exits_open, a.seed)
        assert np.allclose(T_inf, res["T_m0_s"].values[idx]), "infinite capacity must reproduce M0"
        for j in random.Random(1).sample(range(len(idx)), min(5, len(idx))):
            T1, _ = simulate(starts[j:j + 1], rows[j:j + 1], nxt, cap, exits_open, a.seed)
            assert abs(T1[0] - res["T_m0_s"].values[idx][j]) < 1e-9, "one person must reproduce M0"
        print(f"[{name}] verify OK: infinite capacity and single-person runs reproduce M0")

    node_tab = pd.DataFrame.from_dict(stats, orient="index").rename_axis("node_id").reset_index()
    node_tab["cap_per_s"] = node_tab["node_id"].map(cap)
    node_tab = node_tab.sort_values("total_wait", ascending=False)
    res.to_csv(os.path.join(a.out, f"m1_{name}_seat_times.csv"), index=False)
    node_tab.to_csv(os.path.join(a.out, f"m1_{name}_nodes.csv"), index=False)
    return res, node_tab


def load_data(a):
    """Workbook (.xlsx) or three csv files; csvs are packed into a temporary workbook so that
    m0.load applies exactly the same cleaning as in M0."""
    if a.xlsx and a.xlsx.lower().endswith((".xlsx", ".xlsm")):
        return m0.load(a.xlsx)
    if not (a.seats and a.nodes and a.edges):
        sys.exit("give an .xlsx workbook, or --seats seats.csv --nodes nodes.csv --edges edges.csv")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "graph.xlsx")
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            for name, f in (("seats", a.seats), ("nodes", a.nodes), ("edges", a.edges)):
                pd.read_csv(f).to_excel(w, sheet_name=name, index=False)
        return m0.load(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx", nargs="?", help="workbook with sheets seats / nodes / edges")
    ap.add_argument("--seats", help="seats csv (instead of a workbook)")
    ap.add_argument("--nodes", help="nodes csv")
    ap.add_argument("--edges", help="edges csv")
    ap.add_argument("--out", default="m1_output")
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--exits", nargs="*", help="all exit node ids (default: nodes ending in _Exit)")
    ap.add_argument("--ordinary-exits", nargs="*", help="exits used in an ordinary departure (default: ids containing 'main')")
    ap.add_argument("--cases", nargs="*", default=["ordinary", "emergency"], choices=["ordinary", "emergency"])
    ap.add_argument("--speed-scale", type=float, default=M1["ordinary_speed_scale"], help="ordinary-departure speed factor")
    ap.add_argument("--widths", help="csv with columns node_id,width_m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    seats, nodes, edges = load_data(a)
    adj0, all_nodes = m0.build_graph(nodes, edges, m0.PARAMS, a.skip_missing)
    exits = m0.pick_exits(nodes, adj0, a.exits)
    ordinary = a.ordinary_exits or [e for e in exits if "main" in e.lower()] or exits
    cap, defaulted = capacities(nodes, set(exits), node_widths(nodes, a.widths))
    if defaulted:
        print(f"[warn] {len(defaulted)} nodes use PLACEHOLDER widths (e.g. {defaulted[:5]}); "
              "supply measured widths with --widths or width_* columns")

    cases = {"ordinary": (ordinary, a.speed_scale), "emergency": (exits, 1.0)}
    out = {c: run_case(c, *cases[c], seats, nodes, edges, all_nodes, cap, a) for c in a.cases}

    laid, real = m0.add_layout(next(iter(out.values()))[0], seats)
    tmax = max(r["T_m1_s"].max() for r, _ in out.values())
    dmax = max(r["delay_s"].max() for r, _ in out.values())
    summ = []
    for c, (res, node_tab) in out.items():
        laid[["T_m1_s", "delay_s"]] = res[["T_m1_s", "delay_s"]].values
        plot_map(laid, "T_m1_s", "M1 egress time T_i (s)", f"M1 egress time by seat - {c}",
                 os.path.join(a.out, f"m1_{c}_heatmap.png"), real, 0, tmax)
        plot_map(laid, "delay_s", "congestion delay T_M1 - T_M0 (s)", f"Congestion delay by seat - {c}",
                 os.path.join(a.out, f"m1_{c}_delay.png"), real, 0, dmax, "magma")
        for lvl, g in [("all", res)] + [(l, res[res.level == l]) for l in res.level.unique()]:
            ok = g.dropna(subset=["T_m1_s"])
            summ.append(dict(case=c, group=lvl, seats=len(g), unreachable=len(g) - len(ok),
                             T_s=ok["T_m1_s"].max(), T90_s=ok["T_m1_s"].quantile(0.9),
                             mean_s=ok["T_m1_s"].mean(), T_m0_max_s=ok["T_m0_s"].max()))
        top = node_tab.iloc[0]
        print(f"[{c}] T = {res['T_m1_s'].max():.1f} s, T90 = {res['T_m1_s'].quantile(0.9):.1f} s "
              f"(M0 max {res['T_m0_s'].max():.1f} s); bottleneck: {top['node_id']} "
              f"(total wait {top['total_wait']:.0f} s)")
    pd.DataFrame(summ).to_csv(os.path.join(a.out, "m1_summary.csv"), index=False)
    print(f"wrote results to {a.out}/")


if __name__ == "__main__":
    main()