"""Re-draw nine report figures at print size (readable fonts) from SAVED CSV files only.

Reads:  results/results_v2.csv and results/tables/07_*.csv
Writes: Report/figures/*.png  (same file names as the notebook figures; nothing in figures/ is touched)

No notebook is executed, no model is trained, and no data value is changed: every plotted number comes from the
saved CSVs. The only constant that is not in a CSV is the 4.81% "yes" share of the 24,712 original training rows
(1,188 of 24,712, documented in DECISIONS.md), used for the dashed line in the composition figure.

Run from anywhere:  python Report/make_report_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Report' / 'figures'
TAB = ROOT / 'results' / 'tables'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6.5, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.spines.top': False,
                     'axes.spines.right': False, 'lines.linewidth': 1.1, 'lines.markersize': 3, 'figure.dpi': 100})
STRATS = ['static', 'periodic', 'drift_triggered']
COL = {'static': '#2563EB', 'periodic': '#F97316', 'drift_triggered': '#0D9488'}
LS = {'static': '-', 'periodic': '--', 'drift_triggered': ':'}
MK = {'static': 'o', 'periodic': 's', 'drift_triggered': '^'}
LAB = {'static': 'Static', 'periodic': 'Periodic (every 2 windows)', 'drift_triggered': 'Drift-triggered'}
MODELS = ['Logistic Regression', 'Linear SVM']
SCEN = ['real_replay', 'stationary', 'sudden', 'gradual']
SC_TITLE = {'real_replay': 'Real replay', 'stationary': 'Stationary (synthetic)', 'sudden': 'Sudden shift (synthetic)', 'gradual': 'Gradual shift (synthetic)'}
W = 7.16   # two-column text width in inches

v2 = pd.read_csv(ROOT / 'results' / 'results_v2.csv')
v2['lift'] = v2['pr_auc'] - v2['random_pr_auc']


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved Report/figures/' + name)


def panel(ax, scenario, model, value, band=True):
    sub = v2[(v2.scenario == scenario) & (v2.model == model)]
    g = sub.groupby(['strategy', 'window'])[value]
    mean, lo, hi = g.mean().unstack(0), g.min().unstack(0), g.max().unstack(0)
    retr = sub.groupby(['strategy', 'window'])['retrained'].max().unstack(0)
    if value == 'pr_auc':
        ax.plot(mean.index, sub.groupby('window')['random_pr_auc'].mean(), color='grey', ls=(0, (4, 3)), lw=1.2)
    else:
        ax.axhline(0, color='grey', ls=(0, (4, 3)), lw=1.2)
    for s in STRATS:
        if band and scenario != 'real_replay':
            ax.fill_between(mean.index, lo[s], hi[s], color=COL[s], alpha=0.12, lw=0)
        ax.plot(mean.index, mean[s], LS[s], color=COL[s], marker=MK[s])
        w = retr.index[retr[s].astype(bool)]
        ax.plot(w, mean.loc[w, s], '*', color=COL[s], mec='black', mew=0.5, ms=7, zorder=5)
    ax.set_xticks(range(1, 9)); ax.set_xlabel('Stream window')


def legend_strategies(fig, ncol=5, y=-0.02, zero=False):
    h = [plt.Line2D([], [], color='grey', ls=(0, (4, 3)), lw=1.2)]
    l = ['Random baseline (lift = 0)' if zero else 'Random baseline (share of "yes")']
    for s in STRATS:
        h.append(plt.Line2D([], [], color=COL[s], ls=LS[s], marker=MK[s])); l.append(LAB[s])
    h.append(plt.Line2D([], [], marker='*', color='w', mfc='lightgrey', mec='black', ms=7)); l.append('Retrained before this window')
    fig.legend(h, l, loc='lower center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


# ---- 14 and 16: real replay (PR-AUC and lift) -------------------------------------------------------------------
for value, name, ylab in (('pr_auc', '14_real_replay_pr_auc.png', 'PR-AUC (mean over seeds)'), ('lift', '16_real_replay_lift.png', 'Lift (PR-AUC minus random baseline)')):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35), sharey=True)
    for ax, m in zip(axes, MODELS):
        panel(ax, 'real_replay', m, value); ax.set_title(m)
    axes[0].set_ylabel(ylab)
    legend_strategies(fig, y=-0.09, zero=(value == 'lift'))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, name)

# ---- 17: lift, all scenarios ---------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(W, 3.9))
for r, m in enumerate(MODELS):
    for c, s in enumerate(SCEN):
        panel(axes[r, c], s, m, 'lift')
        axes[r, c].set_title(SC_TITLE[s] if r == 0 else '')
        axes[r, c].set_ylabel(f'{m}\nlift' if c == 0 else '')
        axes[r, c].set_xlabel('Stream window' if r == 1 else '')
legend_strategies(fig, y=-0.03, zero=True)
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, '17_all_scenarios_lift.png')

# ---- 18: alerts vs static performance -----------------------------------------------------------------------------
st = v2[(v2.strategy == 'static') & (v2.model == 'Logistic Regression')]
fig, axes = plt.subplots(2, 4, figsize=(W, 3.3), sharex=True)
for c, s in enumerate(SCEN):
    sub = st[st.scenario == s]; g = sub.groupby('window')
    for ax, col, yl in ((axes[0, c], 'max_psi_trigger_groups', 'Largest PSI, model inputs'), (axes[1, c], 'lift', 'Static model lift')):
        mean = g[col].mean(); ax.plot(mean.index, mean.values, '-o', color=COL['static'])
        if s != 'real_replay': ax.fill_between(mean.index, g[col].min(), g[col].max(), color=COL['static'], alpha=0.15, lw=0)
        ax.set_xticks(range(1, 9)); ax.set_ylabel(yl if c == 0 else '')
    axes[0, c].axhline(0.25, color='crimson', ls='--', lw=1); axes[0, c].set_yscale('log'); axes[0, c].set_title(SC_TITLE[s])
    axes[1, c].axhline(0, color='grey', ls=(0, (4, 3))); axes[1, c].set_xlabel('Stream window')
    cw = sub.change_window.dropna()
    if len(cw):
        for ax in axes[:, c]: ax.axvline(cw.iloc[0] - 0.5, color='black', ls=':', lw=1)
fig.text(0.5, -0.01, 'Red dashed line = alert threshold (PSI 0.25); black dotted line = first shifted window; top row on a log scale', ha='center', fontsize=6.5)
fig.tight_layout(); save(fig, '18_alerts_vs_static_performance.png')

# ---- 19: yes share vs PR-AUC (real replay, static Logistic Regression) ----------------------------------------------
sh = pd.read_csv(TAB / '07_A5_yes_share_vs_pr_auc.csv')
yc, pc = sh.columns[1], 'static_LR_PR_AUC'
fig, axes = plt.subplots(1, 3, figsize=(W, 2.15))
ax = axes[0]; ax.bar(sh.window, sh[yc], color='lightgrey', label='Share of "yes" (= random baseline)'); ax.plot(sh.window, sh[pc], '-o', color=COL['static'], label='Static LR PR-AUC')
ax.set_xlabel('Stream window'); ax.set_ylabel('Value'); ax.set_xticks(range(1, 9)); ax.set_ylim(0, 1.0); ax.legend(frameon=False, loc='upper left'); ax.set_title('Share of "yes" and PR-AUC')
ax = axes[1]; ax.plot(sh.window, sh.lift, '-o', color=COL['static']); ax.axhline(0, color='grey', ls=(0, (4, 3))); ax.set_xlabel('Stream window'); ax.set_ylabel('Lift'); ax.set_xticks(range(1, 9)); ax.set_title('Lift above the random baseline')
ax = axes[2]; ax.scatter(sh[yc], sh[pc], color=COL['static'], s=10)
for w_, x_, y_ in zip(sh.window, sh[yc], sh[pc]): ax.annotate(str(int(w_)), (x_, y_), textcoords='offset points', xytext=(3, 3), fontsize=6)
ax.plot([0, 0.75], [0, 0.75], color='grey', ls=(0, (4, 3)), label='PR-AUC = share of "yes"'); ax.set_xlabel('Share of "yes"'); ax.set_ylabel('Static LR PR-AUC'); ax.set_xlim(0, 0.8); ax.set_ylim(0, 1.0); ax.legend(frameon=False, loc='upper left'); ax.set_title('Each point is a window')
fig.tight_layout(); save(fig, '19_real_replay_yes_share_vs_pr_auc.png')

# ---- 20: within-window cross-validation diagnostic ------------------------------------------------------------------
cv = pd.read_csv(TAB / '07_B_oracle_ceiling.csv')
fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), sharey=True)
for ax, m in zip(axes, MODELS):
    c = cv[cv.model == m].sort_values('window')
    ax.plot(c.window, c.random_line, color='grey', ls=(0, (4, 3)), lw=1.2)
    ax.fill_between(c.window, c.oracle_mean - c.oracle_sd_over_folds, c.oracle_mean + c.oracle_sd_over_folds, color='crimson', alpha=0.15, lw=0)
    ax.plot(c.window, c.oracle_mean, '-D', color='crimson')
    for s in STRATS: ax.plot(c.window, c[s], LS[s], color=COL[s], marker=MK[s])
    ax.set_xlabel('Stream window'); ax.set_xticks(range(1, 9)); ax.set_title(m)
axes[0].set_ylabel('PR-AUC on held-out fold rows')
h = [plt.Line2D([], [], color='grey', ls=(0, (4, 3)), lw=1.2), plt.Line2D([], [], color='crimson', marker='D')] + [plt.Line2D([], [], color=COL[s], ls=LS[s], marker=MK[s]) for s in STRATS]
l = ['Random baseline', 'Within-window 5-fold CV (about 800 training rows per fold; mean +/- 1 SD)', 'Static (24,712+ rows)', 'Periodic', 'Drift-triggered']
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Diagnostic using labels unavailable at prediction time; not a performance ceiling', fontsize=7.5, y=1.0)
fig.tight_layout(rect=(0, 0.1, 1, 0.96)); save(fig, '20_oracle_ceiling.png')

# ---- 21: bootstrap intervals ---------------------------------------------------------------------------------------------
bt = pd.read_csv(TAB / '07_C_bootstrap_intervals.csv')
fig, axes = plt.subplots(1, 2, figsize=(W, 3.5))
for ax, cmpn in zip(axes, ['periodic minus static', 'drift_triggered minus static']):
    ylabels, y = [], 0
    for s in SCEN:
        for m in MODELS:
            for k, seed in enumerate([42, 43, 44]):
                r = bt[(bt.scenario == s) & (bt.model == m) & (bt.seed == seed) & (bt.comparison == cmpn)].iloc[0]
                ax.plot([r.ci95_low, r.ci95_high], [y - 0.2 + 0.2 * k] * 2, color=COL['periodic' if cmpn.startswith('periodic') else 'drift_triggered'], lw=1.4)
                ax.plot(r.observed_mean_difference, y - 0.2 + 0.2 * k, 'o', color='black', ms=2)
            ylabels.append(f'{SC_TITLE[s].split(" (")[0]} | {"LR" if m.startswith("Log") else "SVM"}'); y += 1
    ax.axvline(0, color='black', lw=0.8); ax.set_yticks(range(y)); ax.set_yticklabels(ylabels if ax is axes[0] else [''] * y); ax.invert_yaxis()
    ax.set_xlabel('Mean PR-AUC difference over 8 windows'); ax.set_title(cmpn.replace('_', '-'))
fig.text(0.5, -0.01, 'Dot = observed difference; bar = 95% bootstrap interval (within-window sampling variation only); 3 bars = 3 seeds (not independent). LR = Logistic Regression.', ha='center', fontsize=6.3)
fig.tight_layout(); save(fig, '21_bootstrap_intervals.png')

# ---- 22: error analysis ------------------------------------------------------------------------------------------------------
er = pd.read_csv(TAB / '07_D_caught_vs_missed_by_group.csv')
a = er[er.variable == 'age_group']; overall = a.caught.sum() / a.subscribers.sum()
fig, axes = plt.subplots(1, 4, figsize=(W, 2.5), gridspec_kw={'width_ratios': [1, 1.5, 1, 1.1]})
titles = {'age_group': 'Age group', 'job': 'Job', 'poutcome': 'Previous outcome', 'repeat_customer': 'Previous contact'}
for ax, var in zip(axes, titles):
    t = er[er.variable == var].sort_values('recall')
    ax.barh(t.group.astype(str), t.recall, color=np.where(t.small_group_under_30, 'lightgrey', COL['static']), height=0.7)
    for yy, n in enumerate(t.subscribers): ax.text(1.04, yy, f'n={n}', va='center', fontsize=5.5)
    ax.axvline(overall, color='crimson', ls='--', lw=1); ax.set_xlim(0, 1.45); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel('Recall'); ax.set_title(titles[var])
fig.text(0.5, -0.02, f'Recall = share of subscribers caught. Red dashed = overall recall {overall:.2f}; grey = under 30 subscribers; n = subscribers. Descriptive only.', ha='center', fontsize=6.3)
fig.tight_layout(); save(fig, '22_error_analysis_recall_by_group.png')

# ---- 23: training-set composition -------------------------------------------------------------------------------------------------
cp = pd.read_csv(TAB / '07_A7_training_set_composition.csv')
yes = v2[(v2.scenario == 'real_replay') & (v2.model == 'Logistic Regression') & (v2.strategy == 'static') & (v2.seed == 42)].sort_values('window')
fig, axes = plt.subplots(1, 2, figsize=(W, 2.2))
ax = axes[0]
for s in ['periodic', 'drift_triggered']:
    c = cp[cp.strategy == s]; ax.plot(c.window, 100 * c.share_of_training_rows_from_stream, LS[s], color=COL[s], marker=MK[s], label=LAB[s])
ax.set_xlabel('Stream window'); ax.set_ylabel('% of training rows from the stream'); ax.set_ylim(0, 100); ax.set_xticks(range(1, 9)); ax.legend(frameon=False); ax.set_title('Stream share of the scoring model\'s training rows')
ax = axes[1]
ax.bar(yes.window, 100 * yes.random_pr_auc, color='lightgrey', label='"yes" share in each stream window')
ax.axhline(100 * 1188 / 24712, color='crimson', ls='--', label='"yes" share in the original training rows (4.8%)')
ax.set_xlabel('Stream window'); ax.set_ylabel('% "yes"'); ax.set_xticks(range(1, 9)); ax.legend(frameon=False, loc='upper left'); ax.set_title('"yes" share: stream vs. original rows'); ax.set_ylim(0, 75)
fig.tight_layout(); save(fig, '23_real_replay_training_set_composition.png')
