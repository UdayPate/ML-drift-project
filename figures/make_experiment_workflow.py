"""Draws figures/09_experiment_workflow.png (16:9, for a presentation slide).

Run from anywhere:   python figures/make_experiment_workflow.py
Solid blue boxes = implemented and verified. Dashed grey boxes = proposed, not implemented yet.
The numbers shown (41,188 rows, PR-AUC 0.1726, reference 0.1107) come from notebook 03.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / '09_experiment_workflow.png'
BLUE, BLUE_FILL, GREY, GREY_FILL, INK, MUTED = '#2a78d6', '#dbe8fa', '#898781', '#f4f3ee', '#0b0b0b', '#52514e'

fig, ax = plt.subplots(figsize=(13.33, 7.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis('off')
fig.patch.set_facecolor('white')


def box(x, y, w, h, text, implemented, size=13):
    """Draw a labelled box; returns its geometry for connecting arrows."""
    style = dict(boxstyle='round,pad=0.25,rounding_size=1.2')
    if implemented:
        patch = FancyBboxPatch((x, y), w, h, fc=BLUE_FILL, ec=BLUE, lw=2.6, **style)
    else:
        patch = FancyBboxPatch((x, y), w, h, fc=GREY_FILL, ec=GREY, lw=2.2, ls=(0, (5, 3)), **style)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK, linespacing=1.3)
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)


def arrow(p0, p1, color=MUTED, ls='-', lw=2.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=22, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


def elbow(x0, y0, x1, y1, y_mid, label, label_x, label_y, ha='left'):
    """Down, sideways, down connector (dotted) with a label."""
    ax.plot([x0, x0, x1], [y0, y_mid, y_mid], color=MUTED, lw=2.2, ls=(0, (2, 2)))
    arrow((x1, y_mid), (x1, y1), ls=(0, (2, 2)))
    ax.text(label_x, label_y, label, fontsize=11.5, color=MUTED, style='italic', ha=ha, va='center', linespacing=1.15)


ax.text(1, 56, 'Experiment workflow and status', fontsize=22, fontweight='bold', color=INK, va='center')

# legend
box(63, 53.2, 5.5, 3.6, '', True); ax.text(69.8, 55.0, 'Implemented and verified', fontsize=12, va='center', color=INK)
box(63, 48.9, 5.5, 3.6, '', False); ax.text(69.8, 50.7, 'Proposed, not implemented yet', fontsize=12, va='center', color=INK)

# ---- lane 1: implemented ----
ax.text(1, 52.6, 'IMPLEMENTED  (notebooks 01-03, src/, 17 unit tests)', fontsize=13, fontweight='bold', color=BLUE, va='center')
raw = box(1, 37.5, 21, 9.5, 'Raw data\n41,188 rows,\noriginal order', True)
split = box(26, 37.5, 21, 9.5, '60 / 20 / 20\nchronological split\n(no shuffling)', True)
model = box(51, 37.5, 21, 9.5, 'Preprocessing +\nLogistic Regression\nfitted on train rows only', True)
dev = box(76, 37.5, 23, 9.5, 'Development evaluation\nPR-AUC 0.1726 vs\n0.1107 random reference', True, size=12.5)
for a, b in [(raw, split), (split, model), (model, dev)]:
    arrow((a['r'] + 0.4, a['cy']), (b['l'] - 0.4, b['cy']))

# ---- lane 2: proposed ----
ax.text(12, 30.0, 'PROPOSED  (future work)', fontsize=13, fontweight='bold', color=MUTED, va='center')
stream = box(1, 11.5, 16, 14, 'Data stream\n4 scenarios,\nsplit into\nwindows of rows', False)
monitor = box(21, 11.5, 17, 14, 'Drift monitoring\nKS + PSI\nper window\n(feature groups\nkept separate)', False, size=12.5)
static = box(43, 21.0, 20, 6, 'Static model', False)
periodic = box(43, 13.5, 20, 6, 'Periodic retraining', False)
triggered = box(43, 6.0, 20, 6, 'Drift-triggered\nretraining', False)
compare = box(68, 7.5, 31, 21.0, 'Comparison\n\nPR-AUC vs. each window\'s\nrandom reference,\nROC-AUC, F1, detection delay,\nfalse alarms, retraining\ncount, compute time', False, size=12.5)
arrow((stream['r'] + 0.4, stream['cy']), (monitor['l'] - 0.4, monitor['cy']))
for s in (static, periodic, triggered):
    arrow((monitor['r'] + 0.4, monitor['cy']), (s['l'] - 0.4, s['cy']))
    arrow((s['r'] + 0.4, s['cy']), (compare['l'] - 0.4, s['cy']))

# connectors from the implemented lane into the proposed lane
elbow(split['cx'], split['b'] - 0.3, stream['cx'], stream['t'] + 0.3, 33.4,
      'reserved stream\n(20%, untouched so far)', split['cx'] + 1.2, 35.4)
elbow(model['cx'], model['b'] - 0.3, 53.0, static['t'] + 0.3, 33.4,
      'initial model, used by\nall three strategies', model['cx'] + 1.2, 35.4)

ax.text(1, 3.2, 'Monitoring alerts drive only drift-triggered retraining; they are logged in every run to measure detection delay and false alarms.',
        fontsize=12, color=MUTED, va='center')
ax.text(1, 0.8, 'Status: baseline only. No drift detection, scenario, retraining or stream results exist yet.',
        fontsize=12.5, color=INK, fontweight='bold', va='center')

fig.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
print('saved', OUT)
