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

## Entry 3 — Step 0.2 finished: `src/` and `tests/` folders
**Date:** 2026-10-26
**Step:** Step 0.2 (folder layout), completed after the cleanup in Entry 2. Step 0.3 was ticked at the same review.
**What was built:** Two empty files: `src/__init__.py` and `tests/.gitkeep`. No code was written. Two items were added under Pending decisions in `DECISIONS.md`: how notebooks import code from `src/`, and whether to add `pytest`.
**Concept behind it (tutor notes):**
- *Package:* a folder with an `__init__.py` is a regular Python package, so shared code in `src/` can later be imported by notebooks (e.g. `from src.settings import SEED`). That lets settings live in one place instead of being copied into notebooks. Modern Python can import a folder without the file, but it is the standard convention.
- *`.gitkeep`:* Git tracks files, not folders, so an empty folder disappears from a fresh clone. An empty placeholder file keeps `tests/` in the repo until real tests exist.
- *Import path:* notebooks run from `notebooks/`, where Python cannot see `src/` yet. This is why the import method is a pending decision for Phase 2.
**Validation/results:** Both files exist with 0 bytes (`ls -la`, `wc -c`). `git status --short --untracked-files=all` showed only `DECISIONS.md` (modified), `PROGRESS.md` (modified), `src/__init__.py` (new) and `tests/.gitkeep` (new). No packages were installed. Not tested: importing from `src/` (nothing to import yet).
**My explanation (student to fill in):**
**What I'm still unsure about (student to fill in):**
**Recall question:** Why does an empty folder need a placeholder file to be shared through Git?
**Next proposed step:** Not started. Step 1.6 in `PROGRESS.md`: leakage review of every feature other than `duration`.

---

## Entry 4 — Step 1.6: leakage review of features other than `duration` (awaiting student review)
**Date:** 2026-10-27
**Step:** Step 1.6 in `PROGRESS.md`. Not ticked until the student reviews it.
**What was built:** `notebooks/02_leakage_review.ipynb`. It quotes the official UCI documentation (`bank-additional-names.txt`) for each of the 19 inputs other than `duration`, runs descriptive checks, and ends with a provisional feature-by-feature table. No model, no data split, and no change to the data.
**Concept behind it (tutor notes):**
- *Data leakage:* training on information that would not exist at prediction time. *Hard leakage* = recorded after or because of the outcome; *timing leakage* = real but not yet recorded or published when needed.
- *The prediction moment matters.* Moment A (before any contact is scheduled) allows only customer, history and market information. Moment B (just before a planned call) also allows the planned channel and date and the number of earlier calls in this campaign. Moment B is the working assumption but is **not yet confirmed or logged as a decision**.
- *Frequency is not availability.* "Quarterly indicator" says how often a series is measured, not when each value was published.
- *Correlation with the target is not proof of leakage.* A feature can predict well legitimately; the question is only whether it was known in time.
**Validation/results:**
- Documentation source: official UCI archive; its `bank-additional-full.csv` has the same SHA-256 as `data/raw/bank-additional-full.csv`. All 19 quoted definitions were checked verbatim against the names file.
- Notebook run from `notebooks/` in a fresh kernel with `jupyter nbconvert --execute --inplace`: 11 of 11 code cells ran, 0 errors.
- `pdays`/`previous`/`poutcome`: consistent apart from 4,110 rows with `previous > 0` but `pdays = 999` (all `poutcome = failure`), which contradicts the documentation's meaning of 999. This is an encoding question, not leakage.
- `campaign`: min 1, max 56. The check "count(k) <= count(k-1)" fails for 10 values of k, so the rows are not a complete log of every call. This does not prove `campaign` is a final total. A customer-matching test gave results no different from shuffled labels, so it was dropped as uninformative. Verdict: **cannot be settled** from the documentation and data.
- Economic features: 10, 26, 26, 316 and 11 distinct values (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`). The two quarterly series change value at Mar/Jun/Sep/Dec in the observed data, not at calendar-quarter starts; the monthly series change at month boundaries. Publication dates are not documented, so availability is **uncertain**.
- Provisional table: 10 features "include (provisional)" (7 customer attributes, `pdays`, `previous`, `poutcome`); 9 "investigate further" (`contact`, `month`, `day_of_week`, `campaign`, and the 5 economic indicators).
**Limits:** the original paper (Moro et al., 2014) was not retrieved, so it may answer the open questions. External release calendars were not consulted. There is no customer ID, so rows cannot be linked to a customer.
**My explanation (student to fill in):**
**What I'm still unsure about (student to fill in):**
**Recall question:** Why does a quarterly economic indicator being available every quarter not prove it was known on the day of a call?
**Next proposed step:** Student review of Step 1.6 and confirmation of the prediction moment. Then Step 1.7 (one-paragraph EDA summary), unless the student prefers to resolve the open questions first.

**Addendum (2026-10-27) — documentation update after the student's review:**
- *Decision recorded:* the student confirmed **Moment B** (predict immediately before a planned call begins); it is logged in `DECISIONS.md`. It defines the prediction time only and does **not** approve ambiguous features. Two new pending items were logged: whether `campaign`, `contact`, `month`, `day_of_week` were known before the modeled call, and when each economic indicator was publicly available.
- *Six groups documented in the notebook:* (1) customer attributes, provisionally available; (2) previous campaign history, provisionally available subject to verifying its historical meaning, with the 4,110 `previous > 0` / `pdays = 999` rows kept as a documented interpretation issue; (3) current contact information (`campaign`, `contact`, `month`, `day_of_week`), unresolved; (4) the five economic indicators, unresolved; (5) `duration`, excluded; (6) `row_order` and reconstructed `period`, analysis metadata only, not predictive inputs.
- *Concept — conservative vs. broader baseline:* a conservative baseline uses only groups 1-2 (10 features), so "no timing leakage" is easy to defend but predictions may be weaker (not measured). A broader baseline adds groups 3-4 (9 more features) and may predict better, but any feature that was not truly known in time would inflate scores invisibly. Because the economic indicators drift most, this choice also changes what the drift experiment tests. The choice has **not** been made; it belongs to the student. A with/without comparison can show how much results depend on a group, but cannot prove whether its features were known in time.
- *Validation:* `02_leakage_review.ipynb` re-run from `notebooks/` in a fresh kernel: 12 of 12 code cells ran, 0 errors. The new check cell confirmed that groups 1-4 cover exactly the 19 reviewed inputs with no overlap (7 + 3 + 4 + 5), and that `row_order` and `period` are absent from the raw CSV. All 19 documentation quotes and the new "substantial improvement" quote were re-checked verbatim against the official names file. The dataset SHA-256 is unchanged. No model was trained and no split was chosen.
- *Status:* Step 1.6 is still unticked in `PROGRESS.md`, awaiting the student's approval.

---

## Entry 5 — Step 1.7: EDA summary (approved and ticked 2026-10-27)
**Date:** 2026-10-27
**Step:** Step 1.7 in `PROGRESS.md`. Not ticked until the student reviews it.
**What was built:** `EDA_SUMMARY.md` at the repository root: one report-ready paragraph, six slide bullets, short method notes, and a table tracing each number to a section of `notebooks/01_EDA.ipynb`. One separately labelled sentence points to `notebooks/02_leakage_review.ipynb` for the Step 1.6 findings. Person 1's notebook was not modified.
**Concept behind it (tutor notes):**
- *Summarise only what was measured.* Each statement is tied to a saved output, so a reader can trace it and nobody has to trust a number from memory.
- *Effect size vs. significance.* With about 41,000 rows the KS p-values print as 0.0 for every tested feature, so the KS statistic (how large the difference is) and PSI carry the information.
- *PSI depends on method choices.* Many bins for the economic indicators are empty in one of the two periods, so their exact PSI values are sensitive to the small floor applied to bin proportions. "Far above 0.25" is the safe statement.
- *Shift is not degradation.* A change in the inputs or in the subscription rate motivates the research question but does not show that any model gets worse, or why.
- *PR-AUC and prevalence.* The random-guessing PR-AUC equals the positive rate. Because the rate differs between early and late rows (5.6% vs. 24.6% on the illustrative cut), PR-AUC values from different windows must be read against their own baselines. No model has been trained yet.
**Validation/results:** A script compared the summary against the saved outputs of `01_EDA.ipynb` (13 of 13 code cells executed, 0 errors; notebook unchanged vs. `HEAD`): shape, imbalance ratio, NaN count, duplicates, `pdays = 999` share, `default` unknown share, early/late yes rates, and all KS/PSI/mean values in the drift table. The class counts (36,548 / 4,640) were recomputed from the raw file because they appear in the notebook only inside a figure. 39 of 40 checks passed at first; the one failure was a rounding artefact of the check itself (`round(4.685, 2)` gives 4.68 in Python), and the unrounded `euribor3m` early mean is 4.6855, so 4.69 is correct. A separate sensitivity check, outside notebook 01, recomputed PSI with different floors: the economic-indicator values changed substantially and stayed above 0.25 in every variant tried (smallest 1.26); `age` and `campaign` have no empty bins and barely moved. That check is mentioned qualitatively in the summary and not counted as a notebook 01 output.
**Validation rerun (after the student's review):** the check was corrected to compare against the notebook's displayed value with a rounding tolerance and, separately, against the unrounded raw data (the `euribor3m` check is now two checks, so 41 checks in total). All 41 passed; `01_EDA.ipynb` is unchanged vs. `HEAD` and the dataset SHA-256 is unchanged. A short "Evidence levels" block (observed / possible / not yet measured) was added to `EDA_SUMMARY.md` to keep those three things visibly separate.
**Limits:** only 7 features were tested for drift, so "customer and campaign features shift little" applies only to `age` and `campaign`. The 70/30 cut is illustrative, not the modeling split. Reconstructed year-month labels were deliberately not used.
**My explanation (student to fill in):**
**What I'm still unsure about (student to fill in):**
**Recall question:** Why can the same model have a different random-guessing PR-AUC baseline in two windows, and what does that mean for comparing its PR-AUC across them?
**Next proposed step:** Student review of Step 1.7. Not started: anything for Person 2's baseline.

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
