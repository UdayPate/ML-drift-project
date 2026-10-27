# Presentation handoff: DATA 245 mid-semester talk

**Deck:** `DATA245_Midterm_Drift_Presentation.pptx` (16:9, editable; 11 main slides + 3 optional backup slides in a separate appendix section).
**Rebuild:** `cd presentation && npm install && node build_deck.js` regenerates the deck *and this file* from `presentation/build_deck.js`. If you edit the deck directly in PowerPoint or Google Slides, edit this file by hand and do not rebuild over your changes.
**Status labels used on slides:** VERIFIED = measured in our notebooks; PRELIMINARY = development-set results; PROPOSED / HYPOTHESIS = future work, not tested.

## 1. Timing and presenters (all editable)

| Slides | Presenter (edit) | Time |
|---|---|---|
| 1 to 4: hook, bank example, why drift matters, research question | Person 4 (opening presenter) | about 3:00 |
| 5: what the data revealed | Person 1 | about 2:00 |
| 6 to 7: split, preprocessing, baseline | Person 2 | about 3:00 |
| 8 to 10: drift, strategies, experiments | Person 3 | about 4:00 |
| 11: takeaways and questions | Closing presenter | about 1:00 |
| Transitions | everyone | about 1:00 |
| **Total** | | **about 14:00** |

Person 2 and Person 3 are currently the same person, so slides 6 to 10 are presented by one speaker (7 minutes in total). Reassign freely.

## 2. Slide-by-slide outline

### Slide 1: Can a Machine Learning Model Become Outdated?
- **Presenter:** Person 4 (opening presenter) [edit]  ·  **Target time:** 0:50
- **Key points:**
  - Open with the question and take a quick show of hands.
  - Introduce the project title, team and course (placeholders to fill in).
- **Figures / visuals:** (no data figure; conceptual graphic built from editable shapes)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Start with the question on the slide and let it sit for a second. 'If a prediction model worked well two years ago, would you trust it to make decisions today?' Ask for a show of hands: who says yes, who says no, who says it depends? Then say: 'Most of you probably said it depends. That is exactly our project.' Briefly introduce the formal title, which is on the bottom left, and say that we study when a model needs to be updated. [Say the team members' names here.] Then move to the bank example.

### Slide 2: Imagine You're Running a Bank
- **Presenter:** Person 4 [edit]  ·  **Target time:** 0:40
- **Key points:**
  - A bank wants to predict who will subscribe to a term deposit, before a planned marketing call.
  - Customer information goes into a model; out comes YES or NO.
  - But customers and the economy change over time.
- **Figures / visuals:** (conceptual flow built from editable shapes)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Picture yourself as a bank. You want to know which customers will say yes to a term deposit, so you only call the promising ones. You have information about each customer, like age, job, education, existing loans and how an earlier campaign went. A machine learning model learns patterns from past customers and gives a yes or no prediction before the call starts. This is our real dataset, the UCI Bank Marketing data. Now the twist at the bottom: customers change, and the economy changes. So what happens to a model that was trained on old data?

### Slide 3: The World Changes. Models Don't Automatically Adapt.
- **Presenter:** Person 4 [edit]  ·  **Target time:** 0:50
- **Key points:**
  - Customers change; the economy changes.
  - A model trained on old data may become less reliable, but not every shift hurts a model.
  - Retrain too rarely: outdated. Retrain too often: wasted time and computing.
  - How do we know when retraining is worth it?
- **Figures / visuals:** (trade-off graphic built from editable shapes)
- **Speaker notes (also in the PowerPoint notes pane):**

  > In real systems, the world keeps moving. Customer behavior can change, and economic conditions can change. A model trained on older data may become less reliable. I want to be careful: not every change hurts a model, sometimes a model handles it fine. The practical problem is a trade-off. If you retrain too rarely, the model can get outdated. If you retrain too often, you waste time and computing resources. So the real question is the one at the bottom: how do we know when retraining is actually worth it?

### Slide 4: When Should We Retrain a Model?
- **Presenter:** Person 4 [edit]  ·  **Target time:** 0:40
- **Key points:**
  - Research question in plain words.
  - Three strategies: static, periodic, drift-triggered.
  - Hypothesis (not a finding): drift-triggered retraining MAY give a better balance of quality and cost.
- **Figures / visuals:** (strategy cards built from editable shapes)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Here is our research question in simple words: can retraining when we detect a meaningful data change give a better balance of prediction quality and retraining cost than never retraining, or retraining on a fixed schedule? We compare three strategies. Static: train once and never retrain. Periodic: retrain on a fixed schedule. Drift-triggered: retrain when a detector sees a meaningful change in the data. Our hypothesis is that drift-triggered retraining may give a good balance. I want to stress that this is a hypothesis, not a result. We have not tested it yet. [Optional: ask the room which strategy they would pick; we will ask again later.]

### Slide 5: What Did Our Bank Data Reveal?
- **Presenter:** Person 1 [edit]  ·  **Target time:** 2:00
- **Key points:**
  - 41,188 rows, 20 input features plus the yes/no target.
  - 4,640 yes and 36,548 no: about 11.3% subscribe (strong class imbalance).
  - Illustrative early-70% vs late-30% comparison: subscription rate 5.6% vs 24.6%. This is not our modeling split.
- **Figures / visuals:** figures/01_target_distribution.png (EDA, Person 1); native chart: subscription rate early vs late rows (values from notebooks/01_EDA.ipynb)
- **Speaker notes (also in the PowerPoint notes pane):**

  > This is the full UCI Bank Marketing file: 41,188 rows, 20 input features and one target, whether the customer subscribed. Only 4,640 customers said yes and 36,548 said no, so about 11.3 percent are yes. That is called class imbalance. It matters because a model can look accurate by just saying no to everyone, so later we judge models with a measure called PR-AUC instead of accuracy. Now look at the right chart. We took the first 70 percent of rows and the last 30 percent of rows and compared them. The subscription rate went from 5.6 percent to 24.6 percent. Please note this 70/30 cut is only for exploration. Our actual modeling split is different and comes next. Also, the file has no date column, so we keep rows in their original order and we do not rely on reconstructed dates.

### Slide 6: Training on the Past, Predicting the Future
- **Presenter:** Person 2 (you) [edit]  ·  **Target time:** 1:20
- **Key points:**
  - Chronological 60/20/20 split: 24,712 train, 8,238 development, 8,238 reserved future stream.
  - 10 pre-call features; call duration excluded.
  - Preprocessing learns only from training rows.
- **Figures / visuals:** (split bar and cards built from editable shapes; values from src/data.py split and notebooks/03_baseline_logreg.ipynb)
- **Speaker notes (also in the PowerPoint notes pane):**

  > We do not split randomly. A random split would let the model see the future, which cannot happen in real life. So we keep the rows in their original order: the first 60 percent, 24,712 rows, is for training. The next 20 percent, 8,238 rows, is our development set where we check the model. The last 20 percent, another 8,238 rows, is a reserved future stream that we have not touched and will only use in the final experiments. For the model we use 10 features that should be known before a call: customer attributes and the outcome of earlier campaigns. We deliberately leave out call duration, because you only know it after the call. Preprocessing means turning categories into yes or no columns and putting numbers on a comparable scale. The important part is that all of those settings are learned from training rows only, so nothing leaks from later data. One honest note: these ten features are included because of documented pre-call assumptions, we have not verified exactly when each was recorded.

### Slide 7: Our First Model: A Reality Check
- **Presenter:** Person 2 (you) [edit]  ·  **Target time:** 1:40
- **Key points:**
  - PR-AUC 0.1726 vs 0.1107 random reference; ROC-AUC 0.5819.
  - At threshold 0.5: 0 of 8,238 predicted YES (precision, recall, F1 all 0). Highest predicted probability about 0.425.
  - Preliminary development results, not a final held-out evaluation. Does not show that drift caused the weak result.
- **Figures / visuals:** figures/08_baseline_pr_curve.png; native 2x2 confusion-matrix inset (full figure: figures/07_baseline_confusion_matrix.png in the appendix)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Here is our first model, a simple Logistic Regression with no tuning. Our main metric is PR-AUC, also called average precision. It is 0.1726, and a model that ranks customers randomly would score 0.1107, which is just the share of yes customers. So the model ranks customers a bit better than random, but not by much. The ROC-AUC is 0.5819, where 0.5 would be random. Now the surprising part: at the usual 0.5 cut-off, the model predicts yes for zero of the 8,238 development customers, so precision, recall and F1 are all zero. You can see this in the small table: 7,326 true no, 912 missed yes. Why? The highest probability the model ever gives is about 0.425, below 0.5. The model learned from training rows where only about 4.8 percent said yes, so its probabilities stay low. Weak ranking and zero recall at 0.5 are two different things. And to be clear: these are preliminary development results, not the final test, and we are not claiming that data drift caused this weak performance.

### Slide 8: What Happens When the Data Changes?
- **Presenter:** Person 3 (you) [edit]  ·  **Target time:** 1:20
- **Key points:**
  - Covariate drift: inputs change. Label/prior drift: share of YES changes. Concept drift: input-outcome relationship changes.
  - EDA observed input shifts and a changing subscription rate. Concept drift has NOT been established.
  - KS and PSI compare distributions; they do not say whether the model got worse.
- **Figures / visuals:** figures/04_early_vs_late_psi.png (EDA, illustrative first 70% vs last 30% of rows)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Data drift is an umbrella word, so let me split it into three kinds. Covariate drift: the inputs change, for example interest rates or customer ages. Label or prior drift: the share of yes answers changes. Concept drift: the relationship between inputs and outcome changes, so the same customer would now behave differently. In our exploration we saw the first two: input distributions differ between earlier and later rows, and the subscription rate went up. We have not shown concept drift, we have not even tested for it. To measure distribution changes we use two tools. KS asks how far apart two distributions are, and PSI asks how big the shift is. The chart on the right is the PSI for the illustrative early versus late comparison: the economic indicators shift the most. Important: KS and PSI measure changes in the data, not whether the model got worse. That link is something our experiments have to test.

### Slide 9: Three Strategies. Which One Will Win?
- **Presenter:** Person 3 (you) [edit]  ·  **Target time:** 1:10
- **Key points:**
  - Static: train once. Periodic: retrain on a schedule. Drift-triggered: retrain when an alert meets our predefined rule.
  - Only drift-triggered retraining uses the alerts.
  - All three are PROPOSED; no experiment has been run.
  - Audience question: which would you choose?
- **Figures / visuals:** (schematic timelines built from editable shapes; not data)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Here are the three strategies side by side, drawn on a timeline of data windows. These pictures are schematic, not real results. Static: we train once at the start and then leave the model alone. Periodic: we retrain on a fixed schedule, say every few windows, whether or not anything changed. Drift-triggered: a detector watches the incoming data, and only when an alert meets our predefined rule do we retrain. Notice that static and periodic ignore the alerts. Only the drift-triggered strategy uses them. Now a quick show of hands: which one do you think gives the best balance of performance and cost? Static, periodic, or drift-triggered? Good, keep your guess in mind. We honestly do not know yet, that is what the experiments will tell us.

### Slide 10: How Will We Test Our Hypothesis?
- **Presenter:** Person 3 (you) [edit]  ·  **Target time:** 1:30
- **Key points:**
  - Four proposed scenarios: real chronological replay, stationary stream, sudden covariate shift, gradual covariate shift.
  - Scenarios 2 to 4 are synthetic, built by resampling or reweighting observed rows.
  - We will measure performance across windows, detection delay, false alarms, retraining events and computing cost, on the same streams for every strategy.
- **Figures / visuals:** (scenario schematics built from editable shapes; full workflow: figures/09_experiment_workflow.png in the appendix)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Here is how we plan to test it. Four scenarios, all proposed. One, real chronological replay: we replay the reserved future rows in their original order, so the data is real but we do not know in advance when or if it changes. Two, a stationary stream where nothing changes, which tells us how many false alarms and needless retrainings each strategy causes. Three, a sudden shift. Four, a gradual shift. The last three are synthetic: we resample or reweight real rows to create a controlled change, so we know exactly when the change happens. The little pictures are schematics, not data. For every strategy on the same streams we will measure prediction quality across time windows, how long the detector takes to notice a change, false alarms, how many retraining events happen, and the computing cost. Nothing here has been run yet, and the synthetic scenarios will never be presented as historical evidence.

### Slide 11: Can a Model Know When It Needs an Update?
- **Presenter:** Closing presenter [edit]  ·  **Target time:** 1:00
- **Key points:**
  - Completed: data exploration, leakage review, chronological split and preprocessing, initial baseline, preliminary drift evidence.
  - Next: drift detectors, controlled and chronological experiments, strategy comparison, cost vs benefit.
  - Takeaway: the goal isn't to retrain more often, it's to understand when retraining is actually useful.
- **Figures / visuals:** (summary built from editable shapes)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Back to our opening question: can a model know when it needs an update? What we have done so far: we explored the data, reviewed which features are really available before a call, built a chronological split and a leakage-safe preprocessing pipeline, trained an honest baseline, and found preliminary evidence that the data changes over time. What comes next: implement the drift detectors, run the controlled and the real chronological experiments, compare the three strategies, and ask whether better predictions justify the retraining cost. If you remember one thing: the goal is not to retrain more often, it is to understand when retraining is actually useful. Thank you, we are happy to take questions.


## 3. Optional appendix (backup slides, not part of the 14 minutes)

Delete these three slides if you prefer; they only exist for questions.

### Slide 12: Backup: The Full Confusion Matrix
- **Presenter:** Person 2 (you) [edit]  ·  **Target time:** only if asked
- **Key points:**
  - Development rows, threshold 0.5, positive class = subscribed.
  - No row reaches 0.5, so there are no predicted positives.
- **Figures / visuals:** figures/07_baseline_confusion_matrix.png
- **Speaker notes (also in the PowerPoint notes pane):**

  > Backup slide in case someone asks to see the full matrix. Rows are what really happened, columns are what the model predicted. 7,326 true no, zero false positives, 912 false negatives, zero true positives. The threshold is 0.5 and the highest predicted probability is about 0.425, so nothing is classified as yes. A different threshold would be a new, separate decision that we have not made.

### Slide 13: Backup: The Proposed Experiment Workflow
- **Presenter:** Person 3 (you) [edit]  ·  **Target time:** only if asked
- **Key points:**
  - Solid blue = implemented and verified. Dashed grey = proposed, not implemented.
- **Figures / visuals:** figures/09_experiment_workflow.png
- **Speaker notes (also in the PowerPoint notes pane):**

  > Backup slide with the whole workflow. The top row is what exists today: raw data, the 60/20/20 chronological split, preprocessing plus Logistic Regression fitted on training rows, and the development evaluation. The bottom row is proposed: a data stream with four scenarios, drift monitoring with KS and PSI per window, the three strategies, and the comparison of performance and cost. Monitoring alerts drive only the drift-triggered strategy; they are logged in every run so we can measure detection delay and false alarms.

### Slide 14: Backup: What We Treat as Known Before the Call
- **Presenter:** Person 2 (you) [edit]  ·  **Target time:** only if asked
- **Key points:**
  - The baseline uses 10 features that are provisionally available before the call.
  - Nine more features have unresolved availability and are not in the baseline.
  - Call duration is excluded.
- **Figures / visuals:** (native table; source: notebooks/02_leakage_review.ipynb and DECISIONS.md)
- **Speaker notes (also in the PowerPoint notes pane):**

  > Backup slide on leakage. Our prediction moment is just before a planned call begins. The customer attributes and the earlier-campaign history are provisionally available, based on the documentation, but we have not verified exactly when they were recorded. The current-contact features and the economic indicators are unresolved, because the documentation does not say whether each value was known before the call, so they are not in the first baseline. Call duration is excluded because it is only known after the call.


## 4. Likely professor questions and answers (backup)

All numbers below are from the verified notebooks; anything not yet done is stated as such.

**Why did you split chronologically instead of randomly?**
Because the model will be used on *future* data. A random split mixes later rows into training, so the model would effectively see the future and its scores would look better than they could be in practice. We keep the original row order (UCI documents the file as ordered by date) and use the first 60% for training, the next 20% for development and the last 20% as a reserved future stream. The file has no date column, so we use row positions, not invented dates.

**Why is the baseline weak?**
We can describe it but have not proven a single cause. Verified facts: PR-AUC is 0.1726 against a random reference of 0.1107, ROC-AUC is 0.5819, and even on its own training rows the model separates yes from no only weakly (ROC-AUC 0.5534). In the training rows 99.47% of customers have no earlier campaign outcome and only 130 rows have `previous > 0`, versus 26.04% of development rows, so the history features carry almost no learnable signal in training. The baseline is also deliberately conservative (10 features, no economic indicators, untuned). We do not claim that drift caused the weak result.

**Why did Logistic Regression predict zero subscribers?**
At the 0.5 threshold, no development row reaches a predicted probability of 0.5: the highest is about 0.425. The model learned from training rows where only 4.81% are yes, so its probabilities stay low, while 11.07% of development rows are yes. The ranking (PR-AUC, ROC-AUC) does not depend on the threshold and is slightly better than random; the zero recall comes from the cut-off. We did not tune the threshold; the project-wide threshold choice is still an open decision.

**Why PR-AUC instead of relying on accuracy?**
Only about 11% of customers subscribe, so a model that says "no" to everyone would still look accurate. For example, on the development rows our zero-positive model is "right" for 7,326 of 8,238 rows (about 89%, derived from the confusion matrix) while finding none of the 912 subscribers. PR-AUC focuses on how well the model finds the positive class and is compared with a random-ranking reference equal to the positive rate (0.1107 on development rows). We also report ROC-AUC, precision, recall and F1, stating the threshold used.

**How do you prevent leakage?**
(1) Call duration is excluded because it is only known after the call. (2) We reviewed every other feature for availability before a planned call; 10 are provisionally available, 9 are unresolved and kept out of the baseline. (3) Preprocessing is fitted on training rows only, and a unit test changes every later row and confirms the fitted preprocessing does not change. (4) The split is chronological. (5) The reserved future stream has not been used for fitting, tuning or choosing anything. (6) Future retraining may only use earlier windows whose labels would already be available.

**How will drift-triggered retraining be evaluated fairly? (proposed)**
Same data streams, same predeclared random seeds, same model type and features for every strategy; retraining uses only earlier windows with available labels and refits preprocessing on exactly those rows; drift thresholds are calibrated on development data only; performance is PR-AUC per window beside that window's random-guess baseline, plus detection delay, false alarms, number of retrains and compute time. We will report cases where retraining does not help. The number of seeds and the statistical test are still to be decided.

**How do we know whether a detected change actually matters?**
KS and PSI measure *changes in the data*, not whether the model got worse, so an alert is only a hint. Our experiments will compare alerts against measured performance in later labelled windows (false-alarm analysis), report effect sizes (KS statistic, PSI) rather than relying on p-values, and monitor customer/campaign features separately from economic indicators. The current EDA shows shifts but does not show that they hurt a model.

**Extra: Have you shown that drift hurts the model?** No. We have observed differences between earlier and later rows and a weak baseline, but a single development window and no retraining comparison cannot show drift as the cause.

**Extra: Why are the economic indicators not in the baseline?** Their availability at call time is unresolved: the documentation gives measurement frequency but not publication dates. Whether to include them is an open decision, and a with/without comparison is planned.

## 5. Source of every number

`notebooks/01_EDA.ipynb` (dataset size, class counts, early/late rates, PSI figure), `notebooks/02_leakage_review.ipynb` (feature availability), `notebooks/03_baseline_logreg.ipynb` (split sizes, baseline metrics, confusion matrix, PR curve), `EDA_SUMMARY.md`, `BASELINE_SUMMARY.md`, `EXPERIMENT_DESIGN.md`.

## 6. Known limits of this deck

- Team member names, instructor and date on slide 1 are placeholders; nothing was invented.
- Slide visuals for the strategies and scenarios are schematic drawings, not data.
- The deck was generated with PptxGenJS. Google Slides imports shapes, text and images; theme fonts and a few spacing details can shift slightly after import, so please re-check each slide once after uploading.
