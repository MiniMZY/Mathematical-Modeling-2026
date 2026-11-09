#!/usr/bin/env python3
"""Stratified draw: 1 seat per stratum, left half only (hall is symmetric).
Orchestra strata: front (rows A-L, concrete) / back (rows M-U, wood) x middle (center block, seat>=100) / side (wedge, seat<100).
Lodge strata: sections 1-4 (hundreds digit of seat number).
'Left' = seat exits via a left-side aisle node."""
import re, sys, pandas as pd
CSV = sys.argv[1] if len(sys.argv) > 1 else 'auditorium_graph_coding_seats_.csv'
import random
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else random.randrange(10**6)
d = pd.read_csv(CSV)
d['half'] = d.node_id.map(lambda n: 'L' if re.fullmatch(r'[A-Z]{1,2}O?L|[A-Z]{2,3}L\d+', n) else 'R')
d = d[d.half == 'L'].copy()
o = d[d.level == 'orchestra'].copy()
o['stratum'] = o.apply(lambda r: ('front' if r.row_label <= 'L' else 'back') + '-' +
                       ('middle' if r.seat_number >= 100 else 'side'), axis=1)
m = d[d.level == 'lodge'].copy()
m['stratum'] = 'lodge-section-' + (m.seat_number // 100).astype(str)
s = pd.concat([o, m])
out = pd.concat([g.sample(1, random_state=SEED).assign(stratum_size=len(g))
                 for _, g in s.groupby('stratum')])
print(f'seed = {SEED}  (rerun with this seed to reproduce)')
print(out[['stratum', 'stratum_size', 'seat_id', 'row_label', 'node_id', 'seats_to_aisle']].to_string(index=False))
