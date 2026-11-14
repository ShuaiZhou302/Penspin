# (vi) Target-Domain Fine-Tuning Baseline Design

Status: completed, using option B (closed-loop oracle); results are in Section 7 and `results/superdex_finetune.json`.

## 1. Official method

Sources: the paper Lessons from Learning to Spin "Pens" (arXiv 2407.18902) and the official code `real/finetune_ppo.py`, `real/agent/ppo_agent.py`, `real/dataset/ppo_dataset.py`, `scripts/finetune_ppo.sh`.

- **Data:** select 15 oracle trajectories longer than 800 steps in simulation and replay them open-loop on the real robot; keep a trajectory only if it rotates more than 2π; store 15 per object, 45 in total. The paper does not report the number of attempts.
- **Training:** starting from the pretrained proprioceptive student, update all network parameters with L1 behavior cloning (the code path that trains only `actor_mlp` is not enabled).
- **Parameters:** Adam, lr 1e-3, batch 512, no weight decay, 3000 epochs, 10% of frames held out for validation, a checkpoint every 100 epochs from epoch 400 on; the code has no model-selection logic. The normalizer stays in train mode and is updated with target-domain data.

## 2. Measured result: open-loop replay fails completely in SuperDex

Conditions: teacher training pool seeds 100/101, correct pen length, μ=1.0, 119 valid attempts in total.

| Metric | Original IG trajectory | SuperDex open-loop replay |
|---|---|---|
| Mean steps | 305 | 31 (about 1.5 s) |
| Rotation ≥ π | 90/119 | **0/119** |
| Rotation ≥ 2π | 88/119 | **0/119** |
| Mean rotation | 10.06 rad | 0.34 rad (max 1.97) |

With the official filter applied strictly, 10/25/50/100 attempts all yield 0 samples, so training is impossible. This result itself must go into the report: the proposal already commits to reporting the number of successes at each budget.

For comparison, closed-loop success counts (evaluation seeds 42–44, 192 episodes in total, μ=1.0):

| Policy | Rotation ≥ π | Rotation ≥ 2π |
|---|---|---|
| Teacher, closed-loop | 54 | 41 (21%) |
| Contact randomization | 8 | 0 |
| f v4 | 8 | 0 |
| B+tactile | 9 | 0 |
| Calibration v4 | 3 | 0 |
| Proprio only | 2 | 0 |

This shows that the teacher fails mainly because it has no feedback, not because the pen cannot be spun in the target domain.

## 3. Choice of data source

| Option | Method | Advantage | Problem |
|---|---|---|---|
| A. Strictly official | Open-loop replay + 2π filter | Exactly as in the paper | 0 samples; only a feasibility result can be reported |
| **B. Target-domain closed-loop oracle (recommended)** | Teacher runs closed-loop in SuperDex, same 2π filter, then BC | At an estimated 21%, 10/25/50/100 attempts give about 2/5/10/21 successful trajectories | Needs target-domain privileged state, which is not available on a real robot; must be labeled as neither zero-shot nor the official open-loop replay |
| C. Relaxed open-loop filter | Keep the open-loop replay segments before the pen drops | Still open-loop | 100 attempts give only about 3000 frames, with almost no finger-gaiting motions; little room for improvement |

**Recommendation:** A as a mandatory reported result, needing no compute; B as the baseline actually trained, with the report stating clearly that it uses more target-domain privileged state than the official method. C is not done.

The advantage of B is that it is the strongest upper bound for this comparison: if even it cannot beat f and the simple baselines, the conclusion is firmer; if it is clearly better, directly tuning the policy on target-domain data is more effective than aligning observations only.

## 4. Option B pipeline

1. **Collection:** the teacher runs closed-loop in SuperDex, μ=1.0, using training pool seeds 100/101, the same initial states as f's pair data, taking the first N valid attempts in dataset order. Evaluation seeds 42–44 are not used. Save the `[noisy_q, target]` frames visible to the student and the teacher actions.
2. **Filtering:** the primary analysis uses rotation ≥ 2π; the sensitivity analysis uses ≥ π. Both thresholds are fixed before seeing B's results.
3. **Base policy:** the main experiment uses the proprio-only student (model 713476), consistent with the paper. The 3 B+tactile training seeds are an optional supplementary experiment.
4. **Fine-tuning:** 3 fine-tuning seeds per budget; other parameters as in Section 1.
5. **Evaluation:** run μ=0.5/1.0/1.5 on seeds 42–44 with the corrected environment, on the same basis as the existing tables.

Estimated cost: each fine-tuning job takes a few minutes to a dozen or so minutes; SuperDex evaluation is 4 budgets × 3 fine-tuning seeds × 3 μ × 3 evaluation seeds, 108 runs in total, about 40 minutes with 8 processes.

## 5. Differences from the official method

| Item | Official | This design | Reason |
|---|---|---|---|
| Data source | Open-loop replay | Closed-loop oracle | Open-loop success count is 0 |
| Action labels | `target - qpos` from logs; processing script not released | Teacher output actions, range [-1, 1] | Same basis as the distillation labels, and the official conversion is unknown |
| Input window | `obs[i:i+3]`, `hist[i:i+30]` paired with `action[i]`; inputs come from frames after the label | Window ends at time t, padded with the reset-time fill | At deployment only past frames exist; avoids using future information |
| History length | Config says 24, data is 30 | 30 | Consistent with existing students |
| Validation split | Random by frame | By whole trajectory | Required by the proposal |
| Too few frames | Cannot iterate when fewer than one batch | Sampling with replacement, number of updates = 3000 × max(1, frames/512) | Needed to train at small budgets |
| Model selection | None in the code | Fixed to the final epoch | Do not select models using target-domain test results |

## 6. Open issues

**Budgets are not on an equal basis.** The current f and calibration use 100 teacher rollouts plus 241 student rollouts (`train_bt` 122, `train_p` 119). When comparing at the proposal's 100 attempts, fine-tuning can use only 100 rollouts, so this is not a fair equal-budget comparison. The report must either state this plainly or add a version of f and calibration trained on teacher rollouts only.

## 7. Results

The base policy is the proprio-only student (model 713476), with only 1 training seed. The table below is rotation reward, the mean over 3 fine-tuning seeds × 3 evaluation seeds; ± is the standard deviation across fine-tuning seeds. The base policy scores 11.7±0.9 at μ=1.0 (Duration 2.08 s).

| Attempts | 2π trajectories | μ=0.5 | μ=1.0 | μ=1.5 | μ=1.0 Duration (s) |
|---|---|---|---|---|---|
| 10 | 2 | 3.7±0.8 | 4.9±1.3 | 4.9±1.6 | 1.56 |
| 25 | 5 | 4.1±0.8 | 5.0±0.4 | 5.2±0.4 | 1.42 |
| 50 | 11 | 4.0±0.2 | 4.7±0.2 | 4.7±1.1 | 1.56 |
| 100 | 22 | 3.9±0.9 | 4.6±0.5 | 4.2±1.0 | 1.42 |

Sensitivity analysis (μ=1.0): with the π filter, the 4 budgets give 6.6–7.0. At budget 100 with the 2π filter, epoch 400 gives 5.2±0.8, and epoch 400 with lr 1e-4 gives 7.6±0.7. The last two were added only after seeing the main results and are used only to judge whether the conclusion is robust.

All fine-tuned policies are below the base policy, and all have 0 successful episodes (≥ 2π). Validation L1 drops from 0.18 to about 0.09, showing the network learned something on the states the teacher visits, yet closed-loop performance got worse; this is consistent with the state distribution shift of doing BC only on oracle states, but this experiment does not directly verify that explanation.

This result only shows that the closed-loop oracle version of fine-tuning does not help here. The official open-loop replay yields no samples in SuperDex, so the paper's original method itself was not tested.
