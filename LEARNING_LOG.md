# LEARNING_LOG.md — My ML Project Learning Journal

**Purpose:** Build a record for exam review, interviews, and picking up where I left off. The agent writes factual step summaries and teaching notes; **I write my own explanation and questions** so the log accurately reflects what I understand.

## How to use this file
After each completed step, the coding agent fills in the factual fields. I fill in the two student reflection fields in my own words, then the agent can correct or clarify my explanation. One step per entry. Do not use invented test results.

---

## Entry 0 — Project setup
**Date:** 2026-10-26
**Step:** Establish tutor-first project instructions and logs.
**What was built:** Five starter Markdown files: `AGENTS.md` (rules for the coding agent), `CLAUDE.md` (tells Claude Code to follow `AGENTS.md`), `PROGRESS.md` (step checklist), `LEARNING_LOG.md`, and `DECISIONS.md`. No ML code has been written in this step.
**Concept behind it (tutor notes):** Reproducible ML projects need consistent working rules, explicit decisions, and a record of why each step matters. The coding agent is a tutor and helper, not the owner of Git commits or research conclusions.
**Validation/results:** Starter files created; no dataset or model has been tested.
**My explanation (student to fill in):** _Write what these three files are for in your own words._
**What I'm still unsure about (student to fill in):** _Add questions here._
**Recall question:** What's the difference between documenting a change and committing a change to Git?
**Next proposed step:** Step 0.2 in `PROGRESS.md`: set up the folder layout, `requirements.txt`, and `.gitignore`. Then Step 1.1: inspect the UCI dataset format and agree on a minimal, leakage-safe data-loading step before writing code.

---

## Entry 1 — Retrospective: Person 1's EDA notebook (`notebooks/01_EDA.ipynb`)
**Date:** 2026-10-26
**Step:** Retrospective record of exploratory data analysis (EDA) that a teammate (Person 1) built quickly for the mid-presentation, outside the one-step tutoring workflow. The agent did not write or test the original notebook; this entry describes what the notebook contains, checked by re-running it during Entry 2.
**What was built:** A notebook that loads `bank-additional-full.csv`, summarises columns and feature groups, counts `unknown` values, duplicates and `pdays = 999`, plots the target imbalance, rebuilds approximate year-month labels from the row order, compares the first 70% of rows with the last 30% (KS test and PSI), plots the correlation of the five economic features, shows why `duration` is leakage, and saves a no-`duration` copy to `data/processed/`. It produces six figures in `figures/`.
**Concept behind it (tutor notes):**
- *Class imbalance:* only a minority of customers say "yes", so a model that always says "no" scores high accuracy while being useless. That is why PR-AUC (precision–recall area under the curve) is the primary metric, compared against the random-guessing PR-AUC, which equals the positive rate.
- *Data drift:* the data a model sees later can differ from what it was trained on. KS (Kolmogorov–Smirnov) asks "are these two samples from different distributions?"; PSI (Population Stability Index) asks "how big is the shift?". With ~41,000 rows, KS p-values become tiny even for small differences, so look at the KS statistic and PSI too.
- *Leakage:* `duration` is only known after the call ends, so it cannot be used to decide whom to call. Keeping it would inflate scores.
- *Caution:* the dataset has no date column. The notebook's year-month labels are reconstructed (each time the month goes backwards, a new year starts). Whether those labels may be used is an open decision in `DECISIONS.md`; windows stay row-index based.
**Validation/results:** Verified with Python's `csv` module: 41,188 data rows, 21 columns, `y` = 36,548 "no" and 4,640 "yes", semicolon-delimited, row order May → Nov. On re-run in this session (Python 3.12.3, pandas 3.1.0): 0 true NaN, 12 duplicate rows, `pdays = 999` in 96.3% of rows, `default` 20.9% "unknown"; yes rate 5.6% in the first 70% of rows vs 24.6% in the last 30%; PSI 11.52 (`emp.var.rate`), 11.32 (`nr.employed`), 8.49 (`euribor3m`), 5.06 (`cons.price.idx`), 4.48 (`cons.conf.idx`), 0.12 (`age`), 0.05 (`campaign`). Not covered by the notebook: no model has been trained, no leakage review of features other than `duration`, no near-constant-column check, no one-paragraph summary.
**My explanation (student to fill in):**
**What I'm still unsure about (student to fill in):**
**Recall question:** Why can a feature that predicts the outcome extremely well, like `duration`, still be a bad feature to train on?
**Next proposed step:** Finish the remaining Phase 1 items (see the progress audit in the Entry 2 report), then Phase 2.

---

## Entry 2 — Cleanup step: bring the EDA notebook and repository back in line with `AGENTS.md`
**Date:** 2026-10-26
**Step:** One explicitly approved cleanup step covering the rest of Step 0.2, the environment part of Step 0.3, and notebook fixes. No `PROGRESS.md` checkbox was ticked.
**What was built:**
- `notebooks/01_EDA.ipynb`: fixed the `charts/` → `figures/` text; added a note that the 70/30 cut is for EDA illustration only (final split still pending); added a note on reading KS p-values next to effect sizes; added `os.makedirs('../data/processed', exist_ok=True)` before saving; made PR-AUC the primary metric in the team notes. Cell 11 (date reconstruction) was deliberately left unchanged. Re-executed top to bottom.
- `requirements.txt` (exact versions, two labelled sections), `.gitattributes`, a README section for the WSL virtual environment and how to run the notebook.
- `DECISIONS.md`: logged "Keep the GitHub repository public" and added one pending item about reconstructed year-month labels.
- A Python 3.12 virtual environment at `~/venvs/ml-drift`, outside the repository.
**Concept behind it (tutor notes):**
- *Reproducibility:* pinning exact package versions (`==`) lets a teammate rebuild the same environment. A notebook that only runs on one machine is not reproducible.
- *Virtual environment:* an isolated folder of packages for one project, so installs do not interfere with other projects.
- *`.gitattributes`:* tells Git how to treat line endings (Windows uses CRLF, Linux uses LF). It keeps them consistent going forward but does not rewrite files Git already tracks, so warnings may continue for those until someone renormalizes.
- *Statistical vs practical significance:* a tiny p-value only says a difference is detectable, not that it is large.
**Validation/results:** Before changing anything, the dataset was checked with the `csv` module (41,188 rows, 4,640 "yes", `;` delimiter, order preserved). The notebook was executed with `jupyter nbconvert --execute --inplace` from `notebooks/` in a fresh kernel: 13 of 13 code cells ran in order, 0 error outputs, and `data/processed/bank_marketing_for_modeling.csv` was created with shape (41188, 22). All six figures were regenerated and compared side by side with the originals: same data, bars, values and trends; the only visible difference is the font (originals look like Inter, new ones use matplotlib's default DejaVu Sans, probably because Inter is not installed in WSL), which changes image sizes slightly. `pip install --dry-run -r requirements.txt` resolved with no conflicts. Not verified: the original notebook's environment (its metadata says Python 3.13); the notebook was not run on Windows; exact image hashes differ, as expected.
**My explanation (student to fill in):**
**What I'm still unsure about (student to fill in):**
**Recall question:** What does `.gitattributes` change, and what does it not change, for files Git already tracks?
**Next proposed step:** Not started. Candidates in `PROGRESS.md` order: review this step and tick 0.2/0.3 if you agree, then the remaining Phase 1 items (e.g. 1.6 leakage review of features other than `duration`).

---

<!-- Copy this block after each completed step. -->
## Entry N — [Step title]
**Date:** YYYY-MM-DD
**Step:**
**What was built:**
**Concept behind it (tutor notes):**
**Validation/results:**
**My explanation (student to fill in):**
**What I'm still unsure about (student to fill in):**
**Recall question:**
**Next proposed step:**
