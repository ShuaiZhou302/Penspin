# Proposal check, observations and improvement plan (2026-11-07)

The reference is `proposal_revision/main_2page.tex` ("Mind the Touch Gap"). Numbers come from RUN_LOG and `results/*.json`. Unless stated otherwise, mean±sd is across 3 training seeds, and SuperDex (SDX) results are zero-shot evaluations at μ=1.0.

## 1. Status of each proposal item

| # | Proposal commitment | Status | Evidence / note |
|---|---|---|---|
| 1 | Short Allegro replay to check control timing and link-level contact logging | Done | Port tests 0 mismatches; two bugs fixed (contact actor key, pen length) |
| 2 | (i) PenSpin proprio-only student | Done, 1 training seed only | IG 14.2, SDX 11.7 |
| 3 | (ii) Raw-contact student | Done | B+tactile, IG 42.9±2.3, SDX 15.7±2.0 |
| 4 | (iii) Contact noise/dropout/delay randomization | Done | rand, SDX 16.6±1.5 |
| 5 | (iv) Per-link threshold/delay calibration | Done, budget 100 only | calib v4, SDX 17.1±0.9 |
| 6 | (v) Learned alignment f | Done | f v4 15.9±1.6; f v5 15.3±0.8 |
| 7 | (vi) PenSpin-style target fine-tuning | Done, but the official version cannot run | Official open-loop replays give 0 successful demos at all 4 budgets; BC on closed-loop oracle demos is below base at every budget |
| 8 | Budgets of 10/25/50/100 attempts for every target-data method | **Not done** | Only fine-tuning ran all 4 budgets; f and calib only 100, plus 241 extra student replays (not equal-attempt) |
| 9 | Metrics: rotations, time to drop, success rate | Done | Rotation reward, Duration; success (≥2π) is 0 for every student and does not separate methods |
| 10 | Metrics: per-link precision/recall/F1 | Done | |
| 11 | Metric: contact onset timing error | **Not done** | |
| 12 | Metric: state divergence | Partial | Open-loop replays survive 31 steps in SDX vs 305 in IG; pen already 0.32 cm apart after step 1. No systematic divergence curve |
| 13 | Vary target friction | Done | μ=0.5/1.0/1.5 |
| 14 | Vary target compliance | **Not done** | |
| 15 | State-matched reset comparison is a diagnostic only, not a deployable method | **Inconsistent with the proposal** | f and calib were trained on step-wise resync (state-matched) pairs, which need privileged state. A deployable open-loop-pair f was never trained |

## 2. Observations

1. **The contact-observation gap is real, and f narrows it.** Without alignment, the F1 between IG contact and SDX contact is only 0.64. f raises it to 0.83 on teacher states and to 0.74–0.78 on student states. Calibration only reaches 0.69.
2. **The dynamics gap is also large.** The teacher, which has privileged state, drops from 136 in IG to 49 in SDX, keeping only 36%. Open-loop replays fail after 31 steps on average in SDX, versus 305 steps in IG.
3. **In IG, most of the B+tactile student's loss comes from observation.** Replacing only its tactile input with f-style tactile input drops the IG reward from 42.9 to 18.3, just 2.6 above the SDX value of 15.7 (observation share s=0.90).
4. **In SDX, the tactile students cannot be told apart.** B+tactile, rand, calibration and f all fall within 15.7–17.1; the differences are smaller than the sd across training seeds, but all are above the proprio-only student (11.7). Tactile input itself helps (about +4–5); the quality of alignment does not show up.
5. **f's error on the student's own states is not the bottleneck.** f v5 raises F1 on f-student states from 0.721 to 0.755, yet the SDX reward does not change (15.3 vs 15.9), and the IG→SDX loss is 9.4 vs 9.7.
6. **Pen pose estimation is not the bottleneck.** For every tactile student, the pen endpoint estimation error in SDX is about 0.8 cm, similar to IG (f 0.8, B+tactile 0.5).
7. **Students and the teacher respond to friction in opposite ways.** The teacher does best at μ=0.5 (76.8), while the students do worst at μ=0.5 (about 13). Students fail in SDX in a different way from the teacher.
8. **Success rate does not separate methods; there may be a floor effect.** No student reaches 2π in SDX. All tactile students cluster around 16, possibly because the same dynamics failure mode pins them together.

## 3. Current status of the proposal's hypothesis

The proposal asks: "Can a small number of target-domain replays improve the contact observations used to train the policy, and thereby help transfer?"

- **At the observation level: holds.** f's F1 is clearly higher than no alignment and calibration (observation 1).
- **At the transfer level: does not hold under default SDX dynamics.** The gain in alignment quality does not turn into a gain in SDX reward (observations 4, 5).
- **The proposal already anticipated this outcome:** the original text says "vary target friction and compliance to see when alignment helps and when it is overwhelmed by residual dynamics error". The current results fall into the latter case. So the proposal does not need to be overturned. The question that remains open is: "Does alignment start to matter when the dynamics gap gets smaller?"

## 4. Improvement plan

| # | Content | In proposal scope? | Cost | Recommendation |
|---|---|---|---|---|
| E1 | SDX contact-compliance sweep: use the teacher to find settings closer to / farther from IG dynamics, then evaluate all students there | Yes (item 14) | CPU closed-loop evals only, no retraining | **Do now** |
| E2 | Offline metrics: equal-attempt budgets 10/25/50/100 for f and calib F1 (with and without student replays), contact onset timing error, state divergence curve | Yes (items 8, 11, 12) | Minutes per f fit | **Do now** |
| E3 | Deployable open-loop-pair f vs resync-pair f | Yes (item 15) | Offline; first check whether open-loop replays log contact | After E2 |
| E4 | Method change: use the same paired replays to align dynamics too (fit IG contact/friction parameters), 2×2 "observation alignment × dynamics alignment" | No, beyond the proposal (which says dynamics stay fixed) | 3 seeds × ~5 h training per arm | **Decide after E1** |

**Why E1 comes first:** it directly answers the question left open in Section 3, and it is an experiment the proposal already committed to. If the f student separates from B+tactile under settings whose dynamics are closer to IG, the report's conclusion is "alignment helps, but only when the dynamics gap is below some level"; if it does not separate under any setting, the conclusion is "in this task, observation alignment is not enough to improve transfer", and E4 becomes more worthwhile.

**How to write the report:** write it as a diagnostic study. f works at the sensor level, but transfer is limited by dynamics, with the chain of evidence given (observations 2–8). Do not frame it as "the method works".

## 5. E1 design (fixed before seeing results)

**Compliance parameters** (SuperDex `ContactParams`, changed for all hand links and the pen at once; defaults come from SuperDex itself):

| Config | Change | Meaning |
|---|---|---|
| C0 | Default (penalty 1e9 Pa/m, smoothing half-width 5 mm, friction falloff 0.01 m/s) | Reproduces existing results |
| C1 | penalty_coefficient 1e8 | Softer |
| C2 | penalty_coefficient 1e10 | Stiffer |
| C3 | penalty_smoothing_half_distance 1 mm | Shorter force onset range, closer to rigid contact |
| C4 | friction_falloff_vel 0.001 m/s | Closer to static friction at low speed |
| C5 | C3 + C4 | Both, closest to PhysX rigid contact with stiction |

**Code:**
- New directory `penspin-superdex-compliance-20261107`; copy the existing `tools/` and add only an optional contact-parameter override to `sdx_env.py`.
- Defaults unchanged; SHA frozen.
- C0 must match the existing teacher JSON exactly, episode by episode; otherwise stop and diagnose.

**Physics checks (not relaxed):**
- States and rewards are all finite.
- For each config, record the maximum penetration depth (the most negative contact-point distance). Configs above 1 mm are marked invalid and do not enter the student stage.
- Joint excess beyond the IG limits is at most 1e-6 rad.

**Stage 1: teacher**
- 6 configs, evaluation seeds 42–44 × 64 episodes, μ=1.0, same noise as the existing teacher evaluation.
- SE is computed from the sd across the 3 evaluation seeds (the teacher has only one model).

**Stage 2: students**
- Config selection:
  - C0;
  - C+, the config with the highest teacher reward among those passing the physics checks; if it is less than 2·SE above C0, record "no setting closer to IG";
  - C−, the config with the lowest teacher reward.
- Evaluate all students: B+tactile, rand, calibration v4, f v4, f v5 (3 training seeds each), plus 1 proprio-only student.

**Decision rule:**

| Case | Reading |
|---|---|
| At C+, f v4 − B+tactile > 2·SE and f v4 not below rand/calib by more than 2·SE | Supports "alignment helps when dynamics are closer" |
| f v4 not separable at both C+ and C0 | Alignment does not help within the tunable compliance range |
| Supplementary: mean student reward vs teacher reward across configs | Shows whether students are held down by the same dynamics factor; not a primary reading |

**Limitations:**
- f was fit to SDX contact labels under C0. Changing compliance also changes the target contact labels themselves, so f is not re-aligned under C+/C−.
- This experiment measures "how a fixed f student performs under different target dynamics", not "re-aligning for each target".
- For each config, the SDX contact on-rate is also reported, to show how much the labels changed.

## 6. Results (2026-11-07 08:30 CST)

### 6.1 E1 stage 1: teacher under different contact compliance

**Code and reproduction:**
- Directory `penspin-superdex-compliance-20261107`, `CODE_SHA256` frozen.
- C0 matches the existing teacher JSON exactly, episode by episode (192/192 episodes).
- Each config: 3 evaluation seeds × 64 episodes, μ=1.0. The teacher scores 136.3 in IG.

| Config | Teacher reward | Duration | ≥2π / 192 | Max penetration | Median penetration | Max joint excess | Episodes over | SDX contact on-rate |
|---|---|---|---|---|---|---|---|---|
| C0 default | 49.2±3.7 | 6.10 s | 41 | 3.05 mm | 2.04 mm | 8.5e-3 rad | 45 | 6.8% |
| C1 softer | 32.7±3.7 | 4.13 s | 23 | 5.80 mm | 3.05 mm | 7.2e-3 | 37 | 6.4% |
| C2 stiffer | 51.9±7.5 | 6.65 s | 41 | 2.10 mm | 1.14 mm | 8.4e-3 | 45 | 7.2% |
| C3 smoothing 1 mm | 52.7±5.6 | 6.86 s | 43 | 2.07 mm | 0.23 mm | 6.9e-3 | 41 | 7.2% |
| C4 stiction | 18.8±1.5 | 3.20 s | 0 | 3.47 mm | 2.45 mm | 1.3e-2 | 65 | 8.8% |
| C5 C3+C4 | 20.3±3.1 | 3.72 s | 1 | 2.52 mm | 0.36 mm | 1.0e-2 | 61 | 8.2% |

**Reading under the Section 5 rules:**
- **Physics checks:** all 6 configs fail (max penetration > 1 mm in every case, joint excess > 1e-6 rad in every case), including the default config C0 used for all existing SDX results. By the rules, no config can enter the student stage, so stage 2 was not run.
- **No C+ exists:** even ignoring the physics checks, no config raises the teacher reward by more than 2·SE. C3 vs C0 is d=+3.4, with 2·SE=7.7.
- **C− is C4:** low-speed stiction drops the teacher from 49 to 19.

**Observations:**
- When low-speed friction is made closer to static friction (C4, C5), the teacher can barely spin the pen.
- Together with the earlier result (the teacher does best at μ=0.5, 76.8), this shows that the teacher trained in IG relies on the pen slipping between the fingers; in this respect, the SDX default config is "less slippery" than IG.
- Students do worse at μ=0.5, so their failure mode differs from the teacher's.

### 6.2 E2a: equal-attempt budgets

Each cell is contact micro F1 (excluding the pen total-contact channel), the mean over 3 f seeds, with sd across seeds ≤ 0.011. The order is val (teacher) / shift (B+tactile) / shift_p (proprio-only) / shift_f (f student). No alignment: 0.636 / 0.646 / 0.631 / 0.625.

| Budget | f, teacher replays only | Calib, teacher only | f + 241 student replays | Calib + student replays |
|---|---|---|---|---|
| 10 | 0.737 / 0.581 / 0.518 / 0.546 | 0.667 / 0.643 / 0.604 / 0.612 | 0.792 / 0.765 / 0.726 / 0.727 | 0.680 / 0.684 / 0.648 / 0.660 |
| 25 | 0.784 / 0.648 / 0.607 / 0.590 | 0.685 / 0.679 / 0.653 / 0.652 | 0.803 / 0.770 / 0.730 / 0.730 | 0.684 / 0.687 / 0.656 / 0.659 |
| 50 | 0.808 / 0.676 / 0.617 / 0.619 | 0.685 / 0.677 / 0.654 / 0.651 | 0.815 / 0.773 / 0.734 / 0.738 | 0.685 / 0.686 / 0.655 / 0.658 |
| 100 | 0.823 / 0.711 / 0.654 / 0.653 | 0.686 / 0.677 / 0.657 / 0.652 | 0.827 / 0.777 / 0.739 / 0.741 | 0.687 / 0.685 / 0.656 / 0.657 |

**Conclusions:**
- **With equal attempts (teacher replays only):** f beats calibration on teacher states at every budget. But on student states, f stays below calibration, and even below no alignment, up to budget 50; only at 100 does it match or slightly exceed calibration (shift 0.711 vs 0.677).
- **Role of student replays:** f's advantage on student states comes mainly from the 241 extra student replays, not from the budget itself.
- **Calibration:** saturates from budget 25 on.

### 6.3 E2b: contact onset timing error

Window ±5 steps (250 ms). In the delay column, positive means the prediction is late.

| Method | Match rate (val / student states) | Mean \|Δt\| ms | Mean delay ms | Extra predicted onsets per true onset |
|---|---|---|---|---|
| No alignment | 0.53 / 0.53–0.55 | 108 / 89–98 | −32 / −19 to −31 | 0.74 / 0.73–0.79 |
| Calib v4 | 0.55 / 0.54–0.55 | 92 / 83–91 | −4 / −4 to +4 | 0.63 / 0.65–0.81 |
| f v4 | 0.46 / 0.38–0.44 | 72 / 81–86 | +15 / +21 to +33 | 0.08 / 0.16–0.19 |
| f v5 | 0.45 / 0.37–0.44 | 73 / 80–82 | +24 / +28 to +31 | 0.08 / 0.13–0.18 |

**Conclusion:** f's high F1 comes mainly from producing almost no false contact onsets. But it misses 55–60% of SDX contact onsets and is late by 15–33 ms on average. Calibration has no systematic delay. Fast regrasping needs exactly this "when does contact happen" signal, which F1 does not reflect.

### 6.4 E2c: state divergence (teacher open-loop replays, val)

| Steps | 1 | 2–5 | 5–10 | 10–20 | 20–50 | 50–100 | >100 |
|---|---|---|---|---|---|---|---|
| Median pen position error | 0.32 cm | 0.41 | 0.55 | 0.67 | 1.68 | 5.95 | 6.14 |
| Fraction with error < 1 cm | 92% | 91% | 90% | 77% | 27% | 0.4% | 0% |

For resync pairs, the median single-step error is 0.18 cm and the p90 is 0.26 cm.

### 6.5 E3: deployable open-loop-pair f

Training labels are switched to SDX contact from open-loop replays; evaluation still uses resync labels. The F1 order is the same as in 6.2.

| Training data | Budget 10 | Budget 100 |
|---|---|---|
| All rows, teacher replays only | 0.107 / 0.232 / 0.377 / 0.294 | 0.119 / 0.233 / 0.428 / 0.316 |
| All rows, + student replays | 0.108 / 0.236 / 0.447 / 0.322 | 0.120 / 0.247 / 0.444 / 0.342 |
| Only rows with pen error < 1 cm, teacher | 0.249 / 0.335 / 0.388 / 0.334 | 0.228 / 0.321 / 0.400 / 0.350 |
| Only rows with pen error < 1 cm, + student | 0.255 / 0.363 / 0.496 / 0.430 | 0.221 / 0.335 / 0.486 / 0.428 |

**Conclusions:**
- **Open-loop pairs cannot train a usable f:** every budget and every filtering choice is far below no alignment (0.63). The reason is that after 20 steps of open-loop replay the pen has already drifted by more than 1 cm (Section 6.4), so the contact labels from the two simulators no longer correspond to the same state.
- **Impact on the proposal:** the proposal's plan of "open-loop pairs, with state matching only as a diagnostic" does not hold in this task; all f results depend on privileged resync pairs. The report must state this.

### 6.6 Taken together

1. **Alignment works only with privileged pairs:** with privileged resync pairs, f works at the sensor level; with deployable open-loop pairs, f cannot be learned.
2. **High F1 does not mean accurate onset timing:** f has high F1, but misses more than half of the contact onsets and is late.
3. **The compliance sweep cannot reach the student stage:** SDX penalty contact has 2–6 mm penetration under every tunable setting and cannot meet the preset 1 mm physics check; nor does any setting make the teacher clearly closer to IG.
4. **The teacher relies heavily on slip, and students fail differently:** the largest factor in the dynamics gap is low-speed friction/slip behavior, but the students' failure mode differs from the teacher's.

### 6.7 Gate decision (2026-11-07, user: keep the gates)

The two gates, 1 mm penetration and 1e-6 rad joint-limit excess, still apply to SuperDex.

**Consequences:**
- So far, none of the SuperDex results (default contact setting C0) satisfy the physics checks. This includes the student comparison, f v5, fine-tuning and the friction sweep.
- These results can only serve as references under "a target simulator that does not satisfy the physics checks", and the report must state this.
- The student stage of the compliance sweep is not run.

**Causes:**
- **Joint-limit excess:** SuperDex joint limits are spring constraints with a default stiffness of 100 N·m/rad. The 0.7 N·m torque limit alone can push a joint about 7e-3 rad past its limit, which matches the measured 8.5e-3 rad.
- **Penetration:** comes from the compliance of penalty contact itself.

**Next step (Section 7):** first look for a SuperDex setting that passes the gates. If one is found, redo the comparison under that setting; if not, follow the proposal's original fallback: report the feasibility results and switch to a target simulator with explicit contact labels.

## 7. Searching for a SuperDex setting that passes the gates

**Code:**
- Directory `penspin-superdex-admissible-20261107`, `CODE_SHA256` frozen.
- Two optional settings added on top of the compliance-sweep version:
  - `SDX_JOINT`: limit stiffness and damping for the 16 joints;
  - `SDX_SUBSTEPS=n`: split each 5 ms physics step into n SuperDex steps with the torque held constant, so the PD control law and the 5 ms update rate are exactly the same as in IG.
- When unset, results match the earlier ones episode by episode (smoke 8/8).

**Smoke (teacher, seed 42, first 8 episodes):**
- Tuning only joint stiffness (≥1e5) together with n≥5 brings joint excess down to 8.75e-6 rad, still not passing the gate.
- Tuning only contact (penalty ≥1e10, smoothing 0.2–1 mm) brings penetration down to 0.03–0.56 mm.
- Both together (limit stiffness 1e6 or 1e8, penalty 1e11, smoothing 0.2 mm): at n=5/10/20, max penetration is 0.29/0.57/0.59 mm and joint excess ≤ 7e-7 rad, **passing both gates**.
- But the teacher reward varies strongly with n: 89.8/68.2/28.6/24.8 for n=1/5/10/20. This shows that at small n the results are dominated by time-discretization error and have not converged.

**Rules fixed before seeing the full results:**
- **Candidate setting A(n):** limit stiffness 1e6, damping 0, penalty 1e11, smoothing 0.2 mm, n ∈ {10, 20, 40}.
- **Teacher evaluation:** seeds 42–44 × 64 episodes, μ=1.0.
- **Admissible condition:** take the smallest n such that:
  1. all 192 episodes of A(n) pass the gates (max penetration ≤ 1 mm, joint excess ≤ 1e-6 rad, finite values);
  2. the teacher rewards of A(n) and A(2n) differ by < 2·SE (SE from the sd across evaluation seeds), i.e. the time step has converged.
- **None found:** if neither n=10 nor n=20 qualifies, record "no SuperDex setting that passes the gates and converges within the available compute", and switch target simulator per the proposal's fallback.
- **Once found:**
  1. first zero-shot evaluate the existing students under that setting (no retraining): B+tactile, rand, calibration v4, f v4, f v5 with 3 training seeds each, plus 1 proprio-only student;
  2. then decide whether to re-collect pairs, refit f and calibration, and retrain students under that setting.

### 7.1 Results (2026-11-07 08:40 CST)

| n | Teacher reward | Duration | ≥2π / 192 | Max penetration | Episodes >1 mm | Max joint excess | Episodes over |
|---|---|---|---|---|---|---|---|
| 10 | 47.2±5.1 | 5.94 s | 48 | 2.55 mm | 2 | 1.6e-5 rad | 2 |
| 20 | 28.7±0.8 | 3.62 s | 29 | 2.68 mm | 3 | 1.2e-4 | 6 |
| 40 | 12.7±0.7 | 1.69 s | 14 | 2.81 mm | 4 | 4.0e-4 | 8 |

**Convergence check:**
- A10 vs A20: d=18.5, 2·SE=5.9.
- A20 vs A40: d=16.0, 2·SE=1.2.
- Neither pair has converged.

**Reading under the rules:**
- All three n values have a few episodes that fail the gates, and the excess grows with n.
- The teacher reward keeps dropping as the step gets finer and does not converge.
- Conclusion: **no SuperDex setting that passes the gates and converges can be found within the available compute.**
- The "passing the gates" in the 8-episode smoke was only due to the small sample.

**Checking my own code for bugs:**
- **Suspicion:** when splitting steps, the external torque is set again at every substep; could it be accumulating?
- **Test:** move the pen outside the hand and have the empty hand track the same target sequence (`test_substeps.py`).
  - The maximum torque is always 0.7 N·m;
  - At most time points the joint difference between different n is only about 1e-4 rad and does not grow with n, so it is not accumulation.
- **But the empty hand has not fully converged either:** n=10 vs n=80 differs by up to 0.17 rad in a single joint at some time points (joint 13, near its upper limit); n=40 vs n=80 still differs by 0.04 rad.
- **Inference:** SuperDex dynamics at a 5 ms step carry significant discretization error, at least near limits and contacts. This is an inference and has not been verified in the SuperDex source code.

**Impact on the whole project:**
- Under the Section 7 rules, SuperDex as a target simulator can only yield feasibility results.
- To run the comparison in the proposal, the target simulator needs to be switched (the proposal's original fallback).

## 8. Checking MuJoCo as the target simulator (2026-11-07)

**Why this is not "MuJoCo to MuJoCo":** the source is Isaac Gym (PhysX); mjlab is not used. So this is still transfer between two different engines.

**Port:**
- Directory `penspin-mujoco-20261107`, official `mujoco` 4.0.0 installed in a separate environment, `CODE_SHA256` frozen.
- The hand uses IG's own URDF directly (`round_tip/allegro_hand_right_fsr_round_dense.urdf`):
  - all 55 collision bodies imported;
  - `link_3.0` has two `<inertial>` blocks; the first is used, following urdfdom;
  - fixed joints kept as separate bodies;
  - in-hand self-collision disabled (the hand's collision filter in IG is −1);
  - hand gravity cancelled with gravcomp.
- Settings matching IG: PD 3/0.1, torque ±0.7 N·m, D term set to 0 on the first substep of each control step, joint friction 0.01, armature 0.001, IG-calibrated limits.
- 32 tactile channels = the norm of the total pen contact force on each link; IG's −1 channel = the total contact force on the pen.
- **Kinematics check:** fingertip positions after reset in 8 episodes match the IG logs (error < 5e-6 m).
- **Evaluation code:** `closed_loop_mj.py` changes only two places relative to the SuperDex version: the environment class and the fingertip readout.

**Smoke (teacher, seed 42, first 8 episodes; IG 102.6, SuperDex default 21.3):**

| Setting | Teacher reward | Max penetration (all) | Max penetration after 1st control step | Max joint excess | Episodes over |
|---|---|---|---|---|---|
| MuJoCo default (solref 0.02) | 87.0 | 3.38 mm | — | 1.8e-4 rad | — |
| Contact/limit tc 0.001–0.002, d 0.99–0.9999, n=10/20 | 37.6–66.1 | 1.27–2.29 mm | — | 2e-5–1.2e-4 | — |
| Contact tc 0.001, limit tc 0.001/0.0005/0.00025, d 0.9999, n=20/40 | 36.5–51.6 | 1.265 mm | 0.23–1.12 mm | 1.1e-5–1.2e-4 | 4/8 |
| Contact tc 0.0005, limit tc 0.00025, n=40 | 32.4 | 1.265 mm | 0.80 mm | 9.0e-6 | 3/8 |

**Two findings:**

1. **The penetration gate is blocked by the IG initial states.**
   - The kinematics match IG exactly, but under MuJoCo geometry the pen already overlaps the hand in the initial states recorded from IG: across 64 initial states the maximum is 2.82 mm, 10 exceed 1 mm, and the median is 0.52 mm.
   - This overlap comes from the IG states; it may be penetration that PhysX itself allows, or a geometry difference between the two sides (PhysX converts cylinders into convex polyhedra). Any target simulator will run into it.
   - Counting only after the first control step, the best settings reach 0.23–0.80 mm.

2. **MuJoCo cannot meet the joint gate.**
   - In IG's own logs the maximum joint excess is only 1e-7 rad (PhysX uses hard constraints), so 1e-6 rad is achievable for PhysX.
   - MuJoCo limits are soft constraints. Even with the stiffest setting, 3–4/8 episodes still exceed the limit by about 1e-5 rad.
   - Estimate: this comes from overshoot when a finger hits its limit, with magnitude roughly velocity × constraint time constant. Reaching 1e-6 rad would need a time constant of about 1e-5 s and a step of about 5e-6 s, i.e. 1000 times the current compute, which is not feasible.
   - In addition, the teacher reward fluctuates between 32 and 52 with the parameters and does not converge stably.

**Conclusion (by the gates):** like SuperDex, MuJoCo cannot pass the 1e-6 rad joint gate. Under these two gates, currently only hard-constraint engines like PhysX can comply; but PhysX is the source engine, so using it as the target turns this into "one engine against itself".

**User decisions needed:**
1. Should the penetration gate be counted only from after the first control step (excluding overlap inherited from the IG initial states)?
2. Does the 1e-6 rad joint gate stay unchanged for soft-constraint engines? If it does, there is no usable target simulator, and the report can only be built mainly on SuperDex feasibility results and diagnostics.
