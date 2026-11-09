#!/usr/bin/env python3
"""
M2 - behavior and knowledge, no guidance (Bailey Hall).  Follows the M2 flowchart and the LaTeX algorithm.

Flowchart                                          Code
  Input seats.csv, nodes.csv, edges.csv            load_data()
  Compute M0 and baseline seat times               build_graph(), m0_potential(), baseline_times()
  Initialize Monte Carlo run counter r = 1         main(): for run in range(R)
  Draw characteristics for all agents              draw_agents()
  Take the agent k with the earliest clock         heap pop in simulate_run()
  Has agent k reached an exit? -> record time      simulate_run()
  Choose next node (knowledge, perceived cost,     choose_next()
     random behavior)
  Queue at next node too long, and re-route?       re-route loop in simulate_run()
  Calculate travel time, update position, clock    simulate_run()
  Any agents left? / r = R? / statistics / output  main()

Model (LaTeX algorithm)
  M0:   f(x)=0 at exits, f(p)=min_(p,q)[w_pq + f(q)],   T_i = t_row_i + f(a_i)
  J_k(q)  = w_pq / m_k + f(q) + w_c * queue(q) + h * [exit_of(q) != habit exit of k]
  J~_k(q) = J_k(q) (1 + eps),  eps ~ N(0, sigma_k^2)  (drawn separately for every option)
  N(p) = out-neighbours of p plus the previous node n_k;  delta_p = |N(p)|
  P_rand(p,q) = omega_k/delta_p                      if q = n_k
                (1 - omega_k/delta_p)/(delta_p - 1)  otherwise          (uniform 1/delta_p if p has no n_k)
  P_k(p,q)    = (1 - kappa_k) P_rand(p,q) + kappa_k 1[q = argmin_{r in K_k(p)} J~_k(r)]
  Re-route: draw a planned node; while queue(planned) > theta and a uniform draw < a_k: draw again.
  After D_max decisions an agent follows its M0 route.  Start time of agent k: tau_k + t_row_k.
  Doors, stair ends and exits are servers with capacity c = width * flow rate; aisles are unlimited.
  queue_n(t) = max(0, F_n - t);  an agent that ARRIVES at n at t_arr is served at max(t_arr, F_n), then F_n += 1/c_n
  (agents are served in order of arrival; the queue an agent sees is the backlog of agents who have already arrived).

Data (CSV, in --data):
  seats.csv  seat_id, level, row_label, seat_number, node_id (aisle), seats_to_aisle
  nodes.csv  node_id, level, node_type, width_inches
  edges.csv  edge_id, node_1_id, node_2_id, walking_distance_inches, edge_type,
             walking_speed_inches_per_s, free_flow_time_s     (edge length / speed = edge walking time)

Usage
  python3 m2_simulation.py --data m2_data --scenario ordinary --runs 30 --out m2_out_ordinary
  python3 m2_simulation.py --data m2_data --validate            # check: kappa=1, sigma=0, no queues  ->  M0
  --param name=value ...  overrides any entry of PARAMS;  --width Main_Exit=240 ...  overrides node widths (inches)
Every number in PARAMS / SCENARIOS is a PLACEHOLDER until you have sources.
"""
import argparse
import heapq
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

INCH = 0.0254  # metres per inch

PARAMS = dict(
    # M0
    door_delay_s=1.0,          # t_d: delay for passing a door node (not applied to exits)
    v_seat_mps=0.6,            # v_seat: sideways speed along a row
    seat_width_in=20.0,        # delta
    row_delay_s=0.0,           # t_d^row
    # behavior distributions (per agent, independent)
    kappa_beta=(5.0, 2.0),     # kappa_k ~ Beta
    sigma_range=(0.05, 0.30),  # sigma_k ~ Uniform
    a_beta=(2.0, 5.0),         # re-route tendency a_k ~ Beta
    omega_beta=(2.0, 5.0),     # willingness to step back omega_k ~ Beta
    speed_mult=(1.0, 0.15),    # m_k ~ Normal(mean, sd) ...
    speed_clip=(0.5, 1.5),     # ... clipped to [lower, upper]
    p_known=0.8,               # chance that an agent knows a given outgoing edge
    p_habit_main=0.8,          # chance that the habit exit is Main_Exit (otherwise a random exit)
    habit_penalty_s=10.0,      # h
    queue_weight=1.0,          # w_c
    queue_threshold_s=3.0,     # theta
    max_reroll=5,              # safety cap on consecutive re-routes at one node
    max_steps=40,              # D_max: decisions before the agent follows its M0 route
    # capacity
    flow_pps_per_m=1.3,        # phi: persons per second per metre of width
    default_width_in=60.0,     # width used where nodes.csv has none
)
SCENARIOS = {                  # reaction time tau_k ~ Exponential(mean)
    "ordinary": dict(tau_mean_s=20.0),   # Q1
    "lecture": dict(tau_mean_s=5.0),     # Q3
}


# ----------------------------------------------------------------------------
# Input and graph
# ----------------------------------------------------------------------------
def load_data(folder):
    seats = pd.read_csv(os.path.join(folder, "seats.csv"))
    nodes = pd.read_csv(os.path.join(folder, "nodes.csv"))
    edges = pd.read_csv(os.path.join(folder, "edges.csv"))
    need = {"walking_distance_inches", "walking_speed_inches_per_s"}
    if not need <= set(edges.columns):
        raise ValueError(f"edges.csv needs columns {sorted(need)}")
    if edges[list(need)].isna().any().any():
        raise ValueError("edges.csv has edges without a length or speed")
    unknown = (set(edges.node_1_id) | set(edges.node_2_id)) - set(nodes.node_id)
    if unknown:
        raise ValueError(f"edges refer to nodes missing from nodes.csv: {sorted(unknown)[:10]}")
    missing = set(seats.node_id) - set(nodes.node_id)
    if missing:
        raise ValueError(f"seats refer to unknown aisle nodes: {sorted(missing)[:10]}")
    return seats, nodes, edges


def build_graph(nodes, edges, P):
    """adj[p] = list of (q, walk_s, delay_s); walk_s = length / speed, delay_s = t_d if q is a door."""
    ntype = dict(zip(nodes.node_id, nodes.node_type))
    exits = [n for n in nodes.node_id if str(n).endswith("_Exit")]
    adj = defaultdict(list)
    for r in edges.itertuples():
        walk = r.walking_distance_inches / r.walking_speed_inches_per_s
        delay = P["door_delay_s"] if (ntype[r.node_2_id] == "door" and r.node_2_id not in exits) else 0.0
        adj[r.node_1_id].append((r.node_2_id, walk, delay))
    return adj, list(nodes.node_id), exits


def topological_order(adj, all_nodes):
    indeg = {n: 0 for n in all_nodes}
    for lst in adj.values():
        for q, *_ in lst:
            indeg[q] += 1
    stack = [n for n, d in indeg.items() if d == 0]
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for q, *_ in adj.get(u, []):
            indeg[q] -= 1
            if indeg[q] == 0:
                stack.append(q)
    if len(order) != len(all_nodes):
        raise ValueError("edge directions contain a cycle")
    return order


def m0_potential(adj, order, exits):
    """f(x): shortest free-flow time from x to an open exit; exit_of(x): which exit."""
    f = {n: math.inf for n in order}
    exit_of = {n: None for n in order}
    for x in exits:
        f[x], exit_of[x] = 0.0, x
    for p in reversed(order):
        for q, walk, delay in adj.get(p, []):
            c = walk + delay + f[q]
            if c < f[p]:
                f[p], exit_of[p] = c, exit_of[q]
    return f, exit_of


def baseline_times(seats, f, exit_of, P):
    res = seats[["seat_id", "level", "row_label", "seat_number", "node_id", "seats_to_aisle"]].copy()
    res["t_row_s"] = res["seats_to_aisle"] * (P["seat_width_in"] * INCH) / P["v_seat_mps"] + P["row_delay_s"]
    res["T0_s"] = res["t_row_s"] + res["node_id"].map(f)
    res["exit0"] = res["node_id"].map(exit_of)
    return res


def service_rates(nodes, exits, P, override=None):
    """Capacity c_n for doors, stair ends and exits (persons/s). Aisles are unlimited (absent)."""
    rate = {}
    for r in nodes.itertuples():
        if r.node_type in ("door", "stairs") or r.node_id in exits:
            w_in = P["default_width_in"] if pd.isna(r.width_inches) else float(r.width_inches)
            w_in = (override or {}).get(r.node_id, w_in)
            rate[r.node_id] = w_in * INCH * P["flow_pps_per_m"]
    return rate


# ----------------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------------
def draw_agents(n, scen, exits, rng, P):
    ag = dict(
        kappa=rng.beta(*P["kappa_beta"], n),
        sigma=rng.uniform(*P["sigma_range"], n),
        a=rng.beta(*P["a_beta"], n),
        omega=rng.beta(*P["omega_beta"], n),
        tau=rng.exponential(scen["tau_mean_s"], n),
        mult=np.clip(rng.normal(*P["speed_mult"], n), *P["speed_clip"]),
    )
    main = "Main_Exit" if "Main_Exit" in exits else exits[0]
    ag["habit"] = [main if rng.random() < P["p_habit_main"] else exits[rng.integers(len(exits))] for _ in range(n)]
    return ag


def simulate_run(adj, f, exit_of, exits, rate, start_node, t_row, ag, rng, P):
    """One Monte Carlo run. All agents are advanced together, always the one with the earliest clock,
    so they share the queues at doors, stairs and exits. Returns each agent's exit time."""
    n = len(start_node)
    exit_set = set(exits)
    W = {(p, q): (walk, delay) for p, lst in adj.items() for q, walk, delay in lst}
    free_at = {x: 0.0 for x in rate}                    # F_n
    known = [dict() for _ in range(n)]                  # K_k(p), drawn when first needed
    prev = [None] * n                                   # n_k
    steps = [0] * n
    T = np.full(n, np.nan)
    heap = [(ag["tau"][k] + t_row[k], k, start_node[k], False) for k in range(n)]   # (clock, agent, node, served)
    heapq.heapify(heap)

    def queue(q, t):
        return max(0.0, free_at[q] - t) if q in free_at else 0.0

    def known_options(k, p):
        if p not in known[k]:
            outs = [q for q, *_ in adj.get(p, []) if np.isfinite(f[q])]
            kn = [q for q in outs if rng.random() < P["p_known"]]
            if not kn and outs:
                kn = [outs[rng.integers(len(outs))]]    # every agent knows at least one way on
            known[k][p] = kn
        return known[k][p]

    def choose_next(k, p, t):
        outs = [q for q, *_ in adj.get(p, []) if np.isfinite(f[q])]
        back = [prev[k]] if prev[k] is not None and prev[k] not in outs and (prev[k], p) in W else []
        N = outs + back                                  # N(p), delta_p = |N(p)|
        if len(N) == 1:
            return N[0]
        d = len(N)
        if prev[k] in N:                                 # modified random choice
            pr = np.array([ag["omega"][k] / d if q == prev[k] else (1 - ag["omega"][k] / d) / (d - 1) for q in N])
        else:
            pr = np.full(d, 1.0 / d)
        kn = [q for q in known_options(k, p) if q in outs]
        if kn:                                           # cost-based part over K_k(p)
            J = np.array([W[(p, q)][0] / ag["mult"][k] + W[(p, q)][1] + f[q]
                          + P["queue_weight"] * queue(q, t)
                          + P["habit_penalty_s"] * (exit_of[q] != ag["habit"][k]) for q in kn])
            Jt = J * np.maximum(0.0, 1.0 + rng.normal(0.0, ag["sigma"][k], len(kn)))
            pr = (1 - ag["kappa"][k]) * pr
            pr[N.index(kn[int(np.argmin(Jt))])] += ag["kappa"][k]
        return N[rng.choice(d, p=pr / pr.sum())]

    while heap:
        t, k, p, served = heapq.heappop(heap)            # agent with the earliest clock
        if not served and p in free_at:                  # arrived at a door / stairs / exit: wait for a free slot
            start = max(t, free_at[p])                   # first come, first served by ARRIVAL time
            free_at[p] = start + 1.0 / rate[p]
            if start > t:
                heapq.heappush(heap, (start, k, p, True))
                continue
            t = start
        if p in exit_set:                                # reached an exit: record evacuation time
            T[k] = t
            continue
        steps[k] += 1
        if steps[k] > P["max_steps"]:                    # D_max reached: follow the M0 route
            q = min((q for q, *_ in adj.get(p, []) if np.isfinite(f[q])),
                    key=lambda q: sum(W[(p, q)]) + f[q], default=None)
        else:
            q = choose_next(k, p, t)
            for _ in range(P["max_reroll"]):             # queue too long, and re-route?
                if q is not None and queue(q, t) > P["queue_threshold_s"] and rng.random() < ag["a"][k]:
                    q = choose_next(k, p, t)
                else:
                    break
        if q is None:
            continue                                     # dead end (data problem): agent never leaves
        walk, delay = W.get((p, q), W.get((q, p)))
        prev[k] = p
        heapq.heappush(heap, (t + walk / ag["mult"][k] + delay, k, q, False))   # update position and clock
    return T


# ----------------------------------------------------------------------------
# Validation, statistics, plots
# ----------------------------------------------------------------------------
def validate(adj, f, exit_of, exits, nodes, base, P, seed):
    """With kappa=1, sigma=0, a=0, p_known=1, h=0, tau=0, m=1 and unlimited capacity M2 must reproduce M0."""
    Q = dict(P, p_known=1.0, habit_penalty_s=0.0, speed_mult=(1.0, 0.0), flow_pps_per_m=1e9, max_steps=10_000)
    rng = np.random.default_rng(seed)
    n = len(base)
    ag = dict(kappa=np.ones(n), sigma=np.zeros(n), a=np.zeros(n), omega=np.zeros(n), tau=np.zeros(n),
              mult=np.ones(n), habit=["Main_Exit"] * n)
    rate = service_rates(nodes, exits, Q)
    T = simulate_run(adj, f, exit_of, exits, rate, base["node_id"].values, base["t_row_s"].values, ag, rng, Q)
    err = np.abs(T - base["T0_s"].values)
    print(f"validation: max |M2 - M0| = {np.nanmax(err):.6f} s, unreachable = {int(np.isnan(T).sum())}")
    return np.nanmax(err) < 1e-6


def summarize(allr):
    rows = []
    for name, g in [("all", allr)] + [(l, allr[allr.level == l]) for l in ("orchestra", "lodge", "balcony")]:
        pr = g.groupby("run")["T_s"].agg(mean="mean", p90=lambda s: s.quantile(0.9), mx="max")
        R = len(pr)
        rows.append(dict(group=name, mean_s=pr["mean"].mean(), mean_ci95=1.96 * pr["mean"].std(ddof=1) / math.sqrt(R),
                         p90_s=pr["p90"].mean(), evac_time_s=pr["mx"].mean(), m0_mean_s=g["T0_s"].mean(),
                         mean_delay_s=(g["T_s"] - g["T0_s"]).mean()))
    return pd.DataFrame(rows)


def heatmap(res, path, label, title):
    """Schematic seat map (rows x seat order), blue = short, red = long. Uses m0_egress helpers if present."""
    try:
        import m0_egress as m0
    except ImportError:
        print("[info] m0_egress.py not found; heat map skipped")
        return
    laid, real = m0.add_layout(res, res)
    m0.plot_heatmap(laid, path, real, label=label, title=title)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="m2_data")
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="ordinary")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="m2_output")
    ap.add_argument("--closed", nargs="*", default=[], help="exit nodes to close")
    ap.add_argument("--width", nargs="*", default=[], help="override node widths, e.g. Main_Exit=240 (inches)")
    ap.add_argument("--param", nargs="*", default=[], help="override PARAMS entries, e.g. p_known=0.5")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    P = dict(PARAMS)
    for kv in a.param:
        k, v = kv.split("=")
        P[k] = type(PARAMS[k])(float(v)) if not isinstance(PARAMS[k], tuple) else tuple(float(x) for x in v.split(","))
    seats, nodes, edges = load_data(a.data)
    adj, all_nodes, exits = build_graph(nodes, edges, P)
    exits = [x for x in exits if x not in set(a.closed)]
    order = topological_order(adj, all_nodes)
    f, exit_of = m0_potential(adj, order, exits)
    base = baseline_times(seats, f, exit_of, P)
    if base["T0_s"].replace(math.inf, np.nan).isna().any():
        sys.exit("some seats cannot reach an exit; check edges.csv")
    print(f"seats={len(seats)} nodes={len(all_nodes)} edges={len(edges)} exits={exits}")
    print(f"M0 baseline: mean {base.T0_s.mean():.1f} s, max {base.T0_s.max():.1f} s ({base.loc[base.T0_s.idxmax(), 'seat_id']})")
    if a.validate:
        sys.exit(0 if validate(adj, f, exit_of, exits, nodes, base, P, a.seed) else "validation FAILED")

    os.makedirs(a.out, exist_ok=True)
    rate = service_rates(nodes, exits, P, {k: float(v) for k, v in (x.split("=") for x in a.width)})
    rng = np.random.default_rng(a.seed)
    scen = SCENARIOS[a.scenario]
    out = []
    for run in range(1, a.runs + 1):                     # Monte Carlo loop, r = 1 ... R
        ag = draw_agents(len(base), scen, exits, rng, P)
        T = simulate_run(adj, f, exit_of, exits, rate, base["node_id"].values, base["t_row_s"].values, ag, rng, P)
        r = base[["seat_id", "level", "T0_s"]].copy()
        r["run"], r["T_s"] = run, T
        out.append(r)
    allr = pd.concat(out)
    stuck = int(allr["T_s"].isna().sum())
    runs = allr.groupby("run")["T_s"].agg(mean_s="mean", evac_time_s="max").reset_index()
    runs.to_csv(os.path.join(a.out, "m2_runs.csv"), index=False)
    summ = summarize(allr)
    summ.to_csv(os.path.join(a.out, "m2_summary.csv"), index=False)
    seat_mean = allr.groupby("seat_id")["T_s"].mean().rename("T_s").reset_index()
    seat_mean.to_csv(os.path.join(a.out, "m2_seat_mean_times.csv"), index=False)
    print(summ.round(1).to_string(index=False))
    print(f"agents that never left (should be 0): {stuck}")
    res = base[["seat_id", "level", "row_label", "seat_number", "node_id"]].merge(seat_mean, on="seat_id")
    heatmap(res, os.path.join(a.out, "m2_heatmap.png"), f"mean M2 egress time (s), {a.runs} runs",
            f"M2 ({a.scenario}) mean egress time by seat (schematic layout, not to scale)")
    print(f"wrote results to {a.out}/")


if __name__ == "__main__":
    main()
