# Transfer Loss Decomposition: Observation Gap vs Dynamics Gap

Status: completed (2026-11-06 01:30 CST). The design was written before the jobs were submitted; results are in Section 5.

## 1. Question

The B+tactile student scores 42.9±2.3 in IG and drops to 15.7±2.0 in SuperDex (μ=1.0). How much of this drop comes from the change in the tactile signal (observation gap), and how much from the change in pen dynamics? If it is mainly dynamics, then an f that only aligns observations cannot help in the first place, which is one explanation for the current result that "f is level with the baselines".

## 2. Method

Keep the IG dynamics unchanged and replace only the tactile signal seen by the B+tactile student with a "SuperDex-style" signal, then evaluate in IG:

- **f channel (primary analysis):** replace tactile with the predictions of f v4 (`f_v4_tps_b100_s0.pt`). f has F1 0.775 on B+tactile student data and is currently the best model of SuperDex tactile.
- **Calibration channel (secondary):** replace tactile with the output of calibration v4 (`calib_v4.npz`, lag 1). Its F1 is lower (val 0.687); it is used to check whether the conclusion depends on f itself.
- **Raw channel (control):** no replacement, rerun in the same repository. It should reproduce the original IG evaluations (758851 / 772992 / 772995), confirming that the code path is the same.

Models: the 3 B+tactile training seeds (`final.pth` from 758850 / 772991 / 772994), each run on evaluation seeds 42–44 × 64 episodes, the same protocol as the existing IG evaluations. Repository `penspin-student-penlen-20261031` (1923a67), environment code unchanged, 3 groups × 3 seeds = 9 evaluation jobs.

## 3. Decision rule (fixed before seeing results)

Let R = raw-channel IG result, X = f-channel IG result, S = SuperDex μ=1.0 result (15.7), each the mean over the 3 training seeds. Observation share:

s = (R − X) / (R − S)

| s | Interpretation |
|---|---|
| < 1/3 | Dynamics gap dominates; observation alignment has limited headroom |
| 1/3–2/3 | Both contribute |
| > 2/3 | Observation gap dominates; the f student's training is the problem |

Second readout: X vs the f-trained student's IG result (25.7±0.5), both under IG dynamics with f-style tactile. If the f student is clearly higher, training with f does help when only the observation differs, so its lost advantage in SuperDex can only be attributed to dynamics.

## 4. Limitations

- f is only an approximation of SuperDex tactile (F1 0.78, not 1.0); its error can push X up or down, so the calibration channel is added as a second estimate.
- This decomposition is under the IG state distribution only; in SuperDex the observation gap and the dynamics gap also amplify each other, so s cannot be taken as an exact proportion.
- Only the B+tactile student was tested, not the contact-randomization student.

## 5. Results

Jobs 802604–802615 all COMPLETED (same code and settings). Logs confirm that the f and calibration channels were loaded. All 9 raw-channel evaluations match the original evaluations exactly (max difference 0.0). Rotation reward is the mean over the 3 training seeds; ± is the standard deviation across training seeds.

| Tactile seen by the B+tactile student | Dynamics | Rotation reward | Duration (s) | Observation share s |
|---|---|---|---|---|
| Raw IG tactile | IG | 42.9±2.2 | 5.51 | — |
| Calibration v4 output | IG | 29.4±0.3 | 4.06 | 0.50 |
| f v4 output | IG | **18.3±0.1** | 2.82 | **0.90** |
| Actual SuperDex tactile | SuperDex μ=1.0 | 15.7±2.0 | 2.46 | — |
| Reference: f student, f output | IG | 25.7±0.5 | 3.65 | — |
| Reference: f student, actual tactile | SuperDex μ=1.0 | 15.9±1.6 | 2.54 | — |

**Readout (rules fixed in Section 3):**

1. **Primary s = 0.90 > 2/3:** the observation gap dominates the B+tactile transfer loss. Swapping only the tactile signal for f's output, with IG dynamics unchanged, drops it from 42.9 to 18.3, within 2.6 of its SuperDex result (15.7). This also independently validates f: its simulated tactile affects the policy almost exactly as the real SuperDex tactile does.
2. **Secondary s = 0.50 (calibration):** both contribute. Calibration is a coarser model of SuperDex tactile (F1 0.69), so the share depends on the sensing model; the primary analysis uses f.
3. **Second readout:** under the same f-style tactile and IG dynamics, the f student (25.7±0.5) is clearly above B+tactile (18.3±0.1), so training with f helps when only the observation differs (+7.4). The f student then loses another 9.8 in SuperDex and ends level with B+tactile.

**Summary:** the proposal's premise holds: the raw-tactile student fails in SuperDex mainly because the tactile signal changes, and f does help at that level. It does not turn into a SuperDex gain because the f student loses another 9.8 in the second stage, which comes from the dynamics gap or from f's error on the f student's own states; this experiment cannot separate the two.

**Caveat:** both students end near 16 in SuperDex, only slightly above the proprio-only student (11.7), so a floor effect is possible: B+tactile is already low under the f channel and has little left to lose, so its small second-stage loss (2.6) does not by itself show it is more robust to dynamics.

## 6. Follow-up: is f accurate on the f student's own states?

Status: design fixed (2026-11-06, written before collection); results are in Section 7.

**Question:** the f student loses another 9.8 from IG (25.7) to SuperDex (15.9). If f predicts especially poorly on the states the f student itself visits, then the tactile patterns it learned in IG do not hold in SuperDex, and this loss counts as f's error. If f is as accurate on these states as on other students' states, the loss more likely comes from dynamics.

**Method:** take the IG recordings for evaluation seeds 42–44, resync them step by step (state-matched) into SuperDex to collect contact pairs (`collect_pairs.py`, `penlen=ig`, μ=1.0), giving the true SuperDex contact at every step. Compare on the states of 3 students:

- f student (recording 783809, evalFB): the object under test;
- B+tactile student (777024, evalBT) and calibration student (783810, evalTD): controls.

For each group, predict with f v4, calibration v4, and no alignment (IG contact used directly), and compute micro F1 (excluding the pen-total contact channel); precision, recall and contact rate are also reported.

**Decision rule:** let F_f = f's F1 on the f student's states, F_o = the lower of f's F1 on the two control students' states.

| F_o − F_f | Interpretation |
|---|---|
| ≥ 0.05 | f is clearly worse on the f student's own states; f error is a likely source of the 9.8 |
| ≤ 0.03 | f is no worse there than elsewhere; the 9.8 more likely comes from dynamics |
| between | Inconclusive |

**Limitation:** the pairs are one-step resyncs on IG states. In the SuperDex closed loop, the f student visits states under SuperDex dynamics, and f's accuracy on those states cannot be measured, so this test can only rule out or support one explanation: "f is already inaccurate on the states it is familiar with".

## 7. Follow-up results

**One deviation from the design:** the B+tactile evaluation recording (777024) was made with the old recorder, which has no per-step velocities, so it cannot be resynced; the collection script stopped on an assertion. The B+tactile control instead uses the seed 103 pair data (777113): it is f's standing held-out test set and was not used in f training, but it is not evaluation seeds 42–44. The f student and the calibration student follow the original design, using evaluation seeds 42–44.

Results are contact micro F1 (excluding the pen-total contact channel), with precision / recall in parentheses. The standard deviation of f's F1 across the 3 evaluation seeds is 0.001 for the f student and 0.024 for the calibration student.

| States from | Paired steps | True contact rate | No alignment | Calibration v4 | f v4 |
|---|---|---|---|---|---|
| f student | 12,784 | 5.1% | 0.639 | 0.659 | **0.721** (0.76 / 0.68) |
| Calibration student | 14,732 | 5.2% | 0.649 | 0.679 | 0.749 (0.78 / 0.72) |
| B+tactile (seed 103) | 6,598 | 5.0% | 0.646 | 0.685 | 0.775 (0.78 / 0.77) |

**By the rule in Section 6:** F_o − F_f = 0.749 − 0.721 = 0.0285 ≤ 0.03, read as "f is no worse on the f student's states; the 9.8 more likely comes from dynamics".

**But this reading is marginal and should not be reported as a firm conclusion:**

- 0.0285 is only 0.0015 below the 0.03 threshold. Against B+tactile alone the gap is 0.054, which would fall in the "f error is a likely source" band.
- No-alignment F1 is similar across the three state sets (0.64–0.65), so the f student's states are not intrinsically harder; f just gains less on them: +0.08 for the f student, +0.10 for the calibration student, +0.13 for B+tactile. f never saw f-student states in training, which is consistent with its lower recall here (0.68, under-reporting contact).
- On calibration-student seed 44, f's F1 is also only 0.722; the seed-to-seed spread is of the same order as this gap.

**Conclusion:** this check cannot rule out f's error on its own student's states. The safe statement is that f degrades somewhat on f-student states (F1 0.03–0.05 lower) and may account for part of the 9.8, with most of it likely from dynamics; the split cannot be resolved here.
