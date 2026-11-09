#!/usr/bin/env python3
"""
M0 - single-person egress time for every seat (Bailey Hall)

Recurrence (directed acyclic graph, edges point aisle -> door -> (stairs) -> exit):

    w_pq   = l_pq / v_kappa(p,q) + t_d * [q is a door node]        edge travel time
    t_row  = d_s * delta / v_seat + t_row_delay                     seat -> aisle
    f(x)   = 0                                  for x in X_open
    f(p)   = min over edges (p,q) of [ w_pq + f(q) ]
    T_i    = t_row_i + f(a_i)                                       egress time of seat i

Input : the workbook (sheets seats / nodes / edges, template column names)
Output: m0_seat_times.csv, m0_summary.csv, m0_heatmap.png (in --out)

Usage : python3 m0_egress.py m2_data --out m0_output          (folder with seats.csv, nodes.csv, edges.csv)
        python3 m0_egress.py auditorium_graph_coding.xlsx --out m0_output   (workbook with sheets seats / nodes / edges)
        add --skip-missing to ignore edges that still have no distance,
        add --closed Left_Exit to close an exit, --verify to cross-check with Dijkstra.

Notes
  * Distances are read from walking_distance_inches (or walking_distance_m if that column exists); in the CSV
    folder format the speed column is walking_speed_inches_per_s.
  * The door delay t_d is applied to door nodes but not to exit nodes (names ending _Exit).
  * walking_speed_mps is used when filled; otherwise the speed for the edge_type in PARAMS is used.
  * Every number in PARAMS is a PLACEHOLDER until you have sources / measurements.
"""
import argparse
import heapq
import math
import os
import sys
from collections import defaultdict

import numpy as np
import openpyxl
import pandas as pd

INCH = 0.0254  # metres per inch

PARAMS = dict(
    speed_mps={"level": 1.3, "ramp": 1.1, "stairs": 0.7, "other": 1.0},  # by edge_type
    v_seat_mps=0.6,         # sidestep speed along a row           (v_seat)
    seat_width_in=20.0,     # seat width delta if seats.width blank (inches)
    door_delay_s=1.0,       # delay for passing a door node        (t_d)
    row_delay_s=0.0,        # delay from seat to aisle             (t_d^row); 0 = no reaction time
)


# ----------------------------------------------------------------------------
# 1. Load and clean the workbook
# ----------------------------------------------------------------------------
def read_sheet(wb, name):
    rows = list(wb[name].iter_rows(values_only=True))
    head = [str(h).strip() if h is not None else "" for h in rows[0]]
    return pd.DataFrame(rows[1:], columns=head).dropna(how="all")


def load_csv(folder):
    """Read seats.csv / nodes.csv / edges.csv (edge length in inches, speed in inches/s) and return the same
    three tables as load()."""
    seats = pd.read_csv(os.path.join(folder, "seats.csv"))
    nodes = pd.read_csv(os.path.join(folder, "nodes.csv"))
    edges = pd.read_csv(os.path.join(folder, "edges.csv"))
    seats["width"] = np.nan if "width" not in seats.columns else seats["width"]
    edges["dist_m"] = edges["walking_distance_inches"] * INCH
    edges["speed"] = edges["walking_speed_inches_per_s"] * INCH
    return seats, nodes, edges


def load(path):
    if os.path.isdir(path):
        return load_csv(path)
    wb = openpyxl.load_workbook(path, data_only=True)
    seats, nodes, edges = (read_sheet(wb, s) for s in ("seats", "nodes", "edges"))

    if "aisle_node_id" in seats.columns and "node_id" not in seats.columns:
        seats = seats.rename(columns={"aisle_node_id": "node_id"})
    seats = seats.dropna(subset=["seat_id"]).copy()
    for c in ("seat_number", "seats_to_aisle", "width"):
        if c in seats.columns:
            seats[c] = pd.to_numeric(seats[c], errors="coerce")
    if "width" not in seats.columns:
        seats["width"] = np.nan

    nodes = nodes.dropna(subset=["node_id"]).copy()
    nodes["node_id"] = nodes["node_id"].astype(str).str.strip()

    edges = edges.dropna(subset=["node_1_id", "node_2_id"]).copy()
    if "walking_distance_m" in edges.columns:
        edges["dist_m"] = pd.to_numeric(edges["walking_distance_m"], errors="coerce")
    elif "walking_distance_inches" in edges.columns:
        edges["dist_m"] = pd.to_numeric(edges["walking_distance_inches"], errors="coerce") * INCH
    else:
        raise ValueError("edges sheet needs walking_distance_inches or walking_distance_m")
    edges["speed"] = pd.to_numeric(edges.get("walking_speed_mps"), errors="coerce")
    return seats, nodes, edges


# ----------------------------------------------------------------------------
# 2. Build the graph  (edge travel time w_pq)
# ----------------------------------------------------------------------------
def build_graph(nodes, edges, P, skip_missing=False):
    ntype = dict(zip(nodes.node_id, nodes.node_type))
    unknown = (set(edges.node_1_id) | set(edges.node_2_id)) - set(ntype)
    if unknown:
        raise ValueError(f"edges refer to nodes missing from the nodes sheet: {sorted(unknown)[:10]}")

    adj, missing = defaultdict(list), []
    for r in edges.itertuples():
        if pd.isna(r.dist_m):
            missing.append(r.edge_id)
            continue
        v_kappa = r.speed if not pd.isna(r.speed) else P["speed_mps"].get(r.edge_type, P["speed_mps"]["other"])
        w = r.dist_m / v_kappa + (P["door_delay_s"] if ntype[r.node_2_id] == "door" and not str(r.node_2_id).endswith("_Exit") else 0.0)
        adj[r.node_1_id].append((r.node_2_id, w))

    if missing and not skip_missing:
        raise ValueError(f"{len(missing)} edges have no distance (e.g. {missing[:5]}). "
                         "Fill them in, or run with --skip-missing.")
    if missing:
        print(f"[warn] skipped {len(missing)} edges without a distance")
    return adj, list(nodes.node_id)


def topological_order(adj, all_nodes):
    """Kahn's algorithm; raises if the edge directions contain a cycle."""
    indeg = {n: 0 for n in all_nodes}
    for u, lst in adj.items():
        for v, _ in lst:
            indeg[v] += 1
    stack = [n for n, d in indeg.items() if d == 0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v, _ in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    if len(order) != len(all_nodes):
        raise ValueError("edge directions contain a cycle; edges must run aisle -> door -> stairs -> exit")
    return order


def pick_exits(nodes, adj, requested=None):
    ids = list(nodes.node_id)
    if requested:
        bad = set(requested) - set(ids)
        if bad:
            raise ValueError(f"unknown exit nodes: {sorted(bad)}")
        return list(requested)
    named = [n for n in ids if n.endswith("_Exit")]
    if named:
        return named
    has_in = {v for lst in adj.values() for v, _ in lst}
    return [n for n in ids if n not in adj and n in has_in]       # sinks with an incoming edge


# ----------------------------------------------------------------------------
# 3. The recurrence f(x)  (one backward pass over the DAG)
# ----------------------------------------------------------------------------
def egress_potential(adj, order, exits_open):
    f = {n: math.inf for n in order}
    exit_of = {n: None for n in order}
    for x in exits_open:
        f[x], exit_of[x] = 0.0, x
    for p in reversed(order):                 # exits first, aisles last
        for q, w in adj.get(p, []):
            c = w + f[q]
            if c < f[p]:
                f[p], exit_of[p] = c, exit_of[q]
    return f, exit_of


def dijkstra_check(adj, order, exits_open):
    """Independent check: multi-source Dijkstra on the reversed graph."""
    radj = defaultdict(list)
    for u, lst in adj.items():
        for v, w in lst:
            radj[v].append((u, w))
    d = {n: math.inf for n in order}
    pq = []
    for x in exits_open:
        d[x] = 0.0
        heapq.heappush(pq, (0.0, x))
    while pq:
        t, u = heapq.heappop(pq)
        if t > d[u]:
            continue
        for v, w in radj.get(u, []):
            if t + w < d[v]:
                d[v] = t + w
                heapq.heappush(pq, (d[v], v))
    return d


# ----------------------------------------------------------------------------
# 4. Seat egress times  T_i = t_row_i + f(a_i)
# ----------------------------------------------------------------------------
def seat_times(seats, f, exit_of, P):
    res = seats[["seat_id", "level", "row_label", "seat_number", "node_id", "seats_to_aisle"]].copy()
    if res["seats_to_aisle"].isna().any():
        raise ValueError("seats_to_aisle is blank for some seats")
    delta_m = seats["width"].fillna(P["seat_width_in"]) * INCH
    res["t_row_s"] = res["seats_to_aisle"] * delta_m / P["v_seat_mps"] + P["row_delay_s"]
    res["f_aisle_s"] = res["node_id"].map(f)
    res["T_s"] = res["t_row_s"] + res["f_aisle_s"]
    res["exit_used"] = res["node_id"].map(exit_of)
    res.loc[~np.isfinite(res["T_s"]), ["T_s", "exit_used"]] = np.nan
    return res


def summarize(res):
    rows = []
    for name, g in [("all", res)] + [(l, res[res.level == l]) for l in res.level.unique()]:
        ok = g.dropna(subset=["T_s"])
        slow = ok.loc[ok["T_s"].idxmax()] if len(ok) else None
        rows.append(dict(group=name, seats=len(g), unreachable=len(g) - len(ok),
                         mean_s=ok["T_s"].mean(), median_s=ok["T_s"].median(),
                         p90_s=ok["T_s"].quantile(0.9), max_s=ok["T_s"].max(),
                         slowest_seat=None if slow is None else slow["seat_id"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. Heat map (uses x_coord/y_coord if the seats sheet has them, else a schematic grid)
# ----------------------------------------------------------------------------
def _x_orchestra(gr):
    n = gr["seat_number"]
    center = gr[n >= 100].sort_values("seat_number")
    left = gr[(n < 100) & (n % 2 == 0)].sort_values("seat_number")    # seat 2 is next to the centre
    right = gr[(n < 100) & (n % 2 == 1)].sort_values("seat_number")   # seat 1 is next to the centre
    half = (len(center) - 1) / 2 if len(center) else 0.0
    x = pd.Series(np.nan, index=gr.index)
    x[center.index] = np.arange(len(center)) - half
    x[left.index] = -(half + 2 + np.arange(len(left)))
    x[right.index] = half + 2 + np.arange(len(right))
    return x


def _x_arc(gr):
    """Lodge / balcony row: stage-left end -> back centre -> stage-right end."""
    row = gr["row_label"].iloc[0]
    h = gr["node_id"].str[len(row)]
    parity = pd.Series(np.where(gr["seat_number"] % 2 == 0, "L", "R"), index=gr.index)   # evens left, odds right
    half = h.where(h.isin(["L", "R"]), parity)                 # centre aisles (C34) carry no side in their name
    g = gr.assign(half=half, sec=(gr["seat_number"] // 100).astype(int))
    left = g[g.half == "L"].sort_values(["sec", "seat_number"])
    right = g[g.half == "R"].assign(k=lambda d: np.where(d.sec == 4, d.seat_number, -d.seat_number))
    right = right.sort_values(["sec", "k"], ascending=[False, True])
    seq = pd.concat([left, right])
    key = (seq.half + seq.sec.astype(str)).tolist()
    pos, raw = [], 0.0
    for i, k in enumerate(key):
        if i:
            raw += 1.0 + (1.0 if k != key[i - 1] else 0.0)            # gap between sections
        pos.append(raw)
    return pd.Series(np.array(pos) - np.mean(pos), index=seq.index)


def add_layout(res, seats):
    res = res.copy()
    if {"x_coord", "y_coord"} <= set(seats.columns) and seats[["x_coord", "y_coord"]].notna().all().all():
        res["x"], res["y"] = pd.to_numeric(seats["x_coord"]).values, pd.to_numeric(seats["y_coord"]).values
        return res, True
    res["x"], res["y"] = np.nan, np.nan
    for level, g in res.groupby("level"):
        for yi, row in enumerate(sorted(g.row_label.unique())):
            gr = g[g.row_label == row]
            res.loc[gr.index, "x"] = (_x_orchestra(gr) if level == "orchestra" else _x_arc(gr)).values
            res.loc[gr.index, "y"] = yi
    return res, False


def plot_heatmap(res, path, real_coords, label="single-person egress time T_i (s)",
                 title="M0 egress time by seat (schematic layout: rows x seat order, not to scale)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = [l for l in ("orchestra", "lodge", "balcony") if l in set(res.level)]
    ok = res.dropna(subset=["T_s"])
    vmin, vmax = ok["T_s"].min(), ok["T_s"].max()
    ratios = [max(res[res.level == l]["y"].nunique(), 3) for l in levels]
    fig, axes = plt.subplots(len(levels), 1, figsize=(12, 2.2 * sum(ratios) / 4 + 1.5),
                             gridspec_kw={"height_ratios": ratios}, squeeze=False)
    sc = None
    for ax, level in zip(axes[:, 0], levels):
        g = res[res.level == level]
        width = g.groupby("y")["x"].count().max()
        size = float(np.clip((12 * 72 / (width + 14)) ** 2 * 0.3, 4, 90))
        sc = ax.scatter(g["x"], g["y"], c=g["T_s"], s=size, marker="s", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        bad = g[g["T_s"].isna()]
        if len(bad):
            ax.scatter(bad["x"], bad["y"], s=size, marker="x", c="black")
        gk = g.dropna(subset=["T_s"])
        if len(gk):
            s = gk.loc[gk["T_s"].idxmax()]
            ax.scatter([s["x"]], [s["y"]], s=size * 4, facecolors="none", edgecolors="black", linewidths=1.8)
            ax.annotate(f"{s['seat_id']}  {s['T_s']:.0f}s", (s["x"], s["y"]), xytext=(8, 8),
                        textcoords="offset points", color="black", fontsize=8, fontweight="bold")
        if not real_coords:
            rows = sorted(g.row_label.unique())
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels(rows, fontsize=7)
            ax.set_xticks([])
        ax.set_title(f"{level}", loc="left", fontsize=10)
        ax.margins(0.02)
    fig.colorbar(sc, ax=axes[:, 0].tolist(), label=label, shrink=0.8)
    if not real_coords:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xlsx")
    ap.add_argument("--out", default="m0_output")
    ap.add_argument("--skip-missing", action="store_true", help="ignore edges with no distance")
    ap.add_argument("--exits", nargs="*", help="exit node ids (default: nodes ending in _Exit)")
    ap.add_argument("--closed", nargs="*", default=[], help="exit node ids to close")
    ap.add_argument("--verify", action="store_true", help="cross-check with multi-source Dijkstra")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    seats, nodes, edges = load(a.xlsx)
    adj, all_nodes = build_graph(nodes, edges, PARAMS, a.skip_missing)
    order = topological_order(adj, all_nodes)
    exits = pick_exits(nodes, adj, a.exits)
    exits_open = [x for x in exits if x not in set(a.closed)]
    if not exits_open:
        sys.exit("no open exits")
    print(f"seats={len(seats)} nodes={len(all_nodes)} edges used={sum(len(v) for v in adj.values())}")
    print(f"exits open: {exits_open}" + (f"  (closed: {a.closed})" if a.closed else ""))

    f, exit_of = egress_potential(adj, order, exits_open)
    if a.verify:
        d = dijkstra_check(adj, order, exits_open)
        bad = [n for n in order if not (f[n] == d[n] or abs(f[n] - d[n]) < 1e-9)]
        assert not bad, f"DP and Dijkstra disagree at {bad[:5]}"
        print("verify: DP matches multi-source Dijkstra")

    missing_nodes = set(seats.node_id) - set(all_nodes)
    if missing_nodes:
        raise ValueError(f"seats refer to unknown aisle nodes: {sorted(missing_nodes)[:10]}")
    res = seat_times(seats, f, exit_of, PARAMS)
    summ = summarize(res)

    res.to_csv(os.path.join(a.out, "m0_seat_times.csv"), index=False)
    summ.to_csv(os.path.join(a.out, "m0_summary.csv"), index=False)
    print(summ.round(1).to_string(index=False))
    print("seats per exit:", res["exit_used"].value_counts().to_dict())
    if res["T_s"].isna().all():
        sys.exit("no seat can reach an open exit - check edge directions and missing distances")

    laid, real = add_layout(res, seats)
    plot_heatmap(laid, os.path.join(a.out, "m0_heatmap.png"), real)
    print(f"wrote {a.out}/m0_seat_times.csv, m0_summary.csv, m0_heatmap.png")


if __name__ == "__main__":
    main()
