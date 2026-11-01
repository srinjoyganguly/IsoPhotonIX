#!/usr/bin/env python3
"""Plot recorded local solves; never creates or substitutes numerical results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report', type=Path, default=Path('docs/verification/public-mode-benchmarks.json'))
parser.add_argument('--output', type=Path, default=Path('docs/verification/public-mode-convergence.png'))
args = parser.parse_args()
report = json.loads(args.report.read_text())
cases = report['cases']
rows = (len(cases)+1)//2
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
figure, axes = plt.subplots(rows, 2, figsize=(12, 3.8*rows), squeeze=False, constrained_layout=True)
for axis, case in zip(axes.flat, cases):
    points=case['refinements']
    axis.plot([p['resolution_cells_per_um'] for p in points],[p['computed_neff'] for p in points], 'o-',color='#1768ac',label='Local MPB')
    label='Analytic reference' if case['reference_kind']=='analytic_reference' else 'Published solver scalar'
    axis.axhline(case['reference_neff'],color='#c66a16',linestyle='--',label=label)
    analytic=case.get('analytic_slab_neff')
    if analytic and case['reference_kind']!='analytic_reference':
        axis.axhline(analytic,color='#38844c',linestyle=':',label='Analytic slab at these inputs')
    axis.set_xscale('log',base=2)
    ticks=[p['resolution_cells_per_um'] for p in points]
    axis.set_xticks(ticks,[str(t) for t in ticks])
    axis.set_xlabel('Resolution (cells/µm)')
    axis.set_ylabel('Effective index n_eff')
    formatter=ScalarFormatter(useOffset=False); formatter.set_scientific(False)
    axis.yaxis.set_major_formatter(formatter)
    title=case['title'].replace(': ', ':\n')
    axis.set_title(title + f"\nFinest reference difference: {100*case['relative_error']:.6g}%", fontsize=10,loc='left')
    axis.legend(loc='best',fontsize=8)
for axis in list(axes.flat)[len(cases):]: axis.set_visible(False)
figure.suptitle('Local waveguide mode benchmarks\nScalar comparisons and mesh sensitivity; no full-device accuracy claim',fontsize=14)
args.output.parent.mkdir(parents=True,exist_ok=True)
figure.savefig(args.output,dpi=180)
svg = args.output.with_suffix('.svg')
figure.savefig(svg)
# Matplotlib emits trailing spaces in path data; keep generated source clean.
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
print(args.output)
