# MuJoCo as the second target simulator (2026-11-07)

Status: design fixed (written before seeing the full MuJoCo results); results are in Section 5.

## 1. Decision and motivation

**Handling of gates (user decision, 2026-11-07):**
- The 1 mm penetration and 1e-6 rad joint-limit excess become physics metrics that must be reported for every target setting, and are no longer pass/fail criteria.
- Finite values remain a hard check; qpos is not clipped and limits are not relaxed.

**Reasons for this change:**
- The joint gate is 4 orders of magnitude smaller than the observation noise (0.01 rad).
- 2–3 mm of penetration corresponds to soft-contact deformation.
- IG's initial states themselves have up to 2.8 mm of overlap.

**The two things that actually need to be guaranteed:**
1. The target setting is fixed before seeing results;
2. The conclusions do not depend on a particular numerical setting.

**Why add MuJoCo:**
- The SuperDex default setting has not converged at a 5 ms step.
- An 8-episode smoke test showed the teacher scores 87.0 under MuJoCo default settings (IG 102.6, SuperDex 21.3), making it a target with a much smaller dynamics gap.
- So it can answer the question the compliance sweep did not: when the dynamics gap gets smaller, does alignment start to help?
- A conclusion counts as reliable only if it holds on two different targets.

## 2. Target setting M (fixed)

- **Version:** MuJoCo 4.0.0, `penspin-mujoco-20261107` (`CODE_SHA256` frozen).
- **Contact and limits:** MuJoCo default parameters (solref 0.02/1, default solimp, default limit solref/solimp).
- **Solver settings:** implicitfast, pyramidal friction cone, condim 3, μ=1.0, pen inertia computed from geometry.
- **Why defaults:** for the same reason SuperDex uses defaults, this is the "out-of-the-box" target. The stiffer parameters tuned in the smoke test serve only as a sensitivity reference.
- **Substep count n:** take the smallest n in {1, 2, 4} satisfying |R(n) − R(2n)| < 2·SE (R is the teacher reward over 192 episodes, SE uses the sd across evaluation seeds).
- **If n=4 still does not converge:** use n=4 anyway, and state the non-convergence in the report.
- **All physics metrics are reported:** penetration over the whole episode, penetration after the first control step, joint-limit excess, contact-on rate.

## 3. Pipeline

1. **Teacher:** seeds 42–44 × 64 episodes, n=1/2/4/8; determine n by the rule above.
2. **Zero-shot students (no retraining):**
   - B+tactile, rand, calibration v4, f v4, f v5, 3 training seeds each, plus 1 proprioception-only student;
   - evaluation seeds 42–44 × 64 episodes;
   - note: f and calibration were fit to SuperDex labels, so under M they serve as a "wrong target" control.
3. **Alignment on M:**
   - collect resync pairs and open-loop pairs in M from the same IG recordings: `collect_pairs` only swaps the environment for MuJoCo;
   - data split identical to v4: train=T100/101, val=T102, shift=BT103, train_bt=BT104/105, train_p=P106/107, shift_p=P108;
   - fit f_M (budget 100, plus train_bt, train_p, f seeds 0/1/2, seed 0 used for students) and calibration calib_M with the same settings.
4. **M-aligned students:**
   - new repo cloned from penlen repo 1923a67, swapping only the f/calibration files;
   - 3 training seeds each for f_M and calib_M, with the same DAgger settings as the f v4 students;
   - pass the port test first (0 mismatches between the in-environment channel and offline predictions), then submit training.
5. **Evaluation:** all students evaluated zero-shot under M, with IG-protocol evaluation as a supplement.

## 4. Decision rule (fixed before seeing results)

**Primary metric:** rotation reward under M at μ=1.0, taken as the mean over 3 training seeds. Let d = mean of A − mean of B, SE = sqrt(sd_A²/3 + sd_B²/3). A counts as higher than B only if d > 2·SE; |d| ≤ 2·SE counts as inseparable.

| Comparison | Question |
|---|---|
| f_M student − B+tactile | With a smaller dynamics gap, does learned alignment beat raw contact? |
| f_M student − calib_M student | Does learned alignment beat simple calibration? |
| f_M student − rand | Does learned alignment beat randomization? |
| Zero-shot tactile students − proprio | Does touch help in M at all? (supplementary) |

**Offline:** F1 of f_M and calib_M on val/shift/shift_p of the M pairs, and contact onset timing error (same method as E2b).

**How the overall conclusion is worded:**
- Write "alignment helps when the dynamics gap is small" only if the f_M student satisfies d > 2·SE against all of B+tactile, calib_M, and rand.
- Otherwise write "on both targets, alignment did not improve the policy".

## 5. Results

### 5.1 Teacher and substep convergence (2026-11-07 09:50 CST)

Teacher under M (μ=1.0), 192 episodes for each n. IG is 136.3, SuperDex default is 49.2.

| n | Per eval seed | Mean±sd | Duration | ≥2π / 192 |
|---|---|---|---|---|
| 1 | 113.2 / 103.3 / 76.7 | 97.7±18.9 | 11.00 s | 92 |
| 2 | 108.2 / 105.5 / 86.9 | 100.2±11.6 | 11.33 s | 95 |
| 4 | 113.3 / 106.1 / 85.1 | 101.5±14.7 | 11.46 s | 95 |
| 8 | 118.2 / 98.6 / 85.8 | 100.8±16.3 | 11.36 s | 97 |

**Convergence:**
- n1 vs n2: d=−2.5, 2·SE=25.6; by the rule, take **n=1**.
- Limitation: the sd across evaluation seeds is large, so this check has low statistical power. But the absolute differences between adjacent n are all ≤ 2.5 (about 2.5%), far smaller than the 18–16 in SuperDex.

**Physics metrics (reported, not gates):**
- Teacher: max penetration 4.57 mm, median 1.97 mm; max after the first control step is also 4.57 mm; max joint-limit excess 2.2e-3 rad; all values finite; contact-on rate 7.5%.
- Students (3,072 episodes): max penetration 6.33 mm, median 1.57 mm; max joint-limit excess 3.1e-3 rad, 653 episodes exceed 1e-6 rad.

### 5.2 Zero-shot students (no retraining)

The students' f/calibration were fit to SuperDex labels; mean±sd is across 3 training seeds. The ≥π count covers all 576 episodes.

| Student | IG | SuperDex | MuJoCo M | M duration | M ≥π / 576 |
|---|---|---|---|---|---|
| B+tactile | 42.9±2.3 | 15.7±2.0 | 30.2±1.5 | 3.82 s | 139 |
| Calib v4 | 35.4±3.7 | 17.1±0.9 | 30.3±1.2 | 3.84 s | 133 |
| rand | 26.1±0.9 | 16.6±1.5 | 27.7±0.9 | 3.68 s | 137 |
| f v4 | 25.7±0.5 | 15.9±1.6 | 26.9±1.1 | 3.60 s | 94 |
| f v5 | 24.7±1.4 | 15.3±0.8 | 24.0±0.9 | 3.18 s | 75 |
| Proprio (1 seed) | 14.2 | 11.7 | 15.2 | 2.64 s | 3 / 192 |

**Observations:**
- In M, students largely keep their IG ranking (B+tactile and calibration highest, f students lowest), whereas in SuperDex all tactile students cluster around 16.
- All tactile students are clearly above the proprioception-only student.
- The f students are low in M mainly because they were already low in IG (f is aligned to SuperDex labels).

### 5.3 Pairs and offline alignment on M

**Pairs:**
- 9 IG recordings, split into two halves 0–32/32–64 as in v4, match whole-episode collection element-wise.
- The dataset has 81,820 rows and matches the v4 dataset element-wise in split, episode, t, IG contact, q, target, and action.

**The gap between the two simulators is clearly smaller:**
- Median resync single-step pen error: M 0.047 cm, SuperDex 0.18 cm.
- Contact-on rate (excluding the pen total contact channel): IG 8.0%, M 7.0%, SuperDex 4.8%.

Contact micro F1 (excluding the pen total contact channel):

| Method | val | shift | shift_p |
|---|---|---|---|
| No alignment | 0.790 | 0.796 | 0.792 |
| calib_M (lag 0) | 0.800 | 0.798 | 0.790 |
| f_M seed 0 (used) / seed 1 / seed 2 | 0.848 / 0.847 / 0.848 | 0.838 / 0.834 / 0.837 | 0.830 / 0.830 / 0.831 |

f_M is about 0.05 above no alignment, whereas in SuperDex f is about 0.19 above. So in M, the contact gap that alignment can correct is itself much smaller.

### 5.4 M-aligned students (in progress)

- **Repo:** `penspin-student-mjalign-20261107`, branch mj, commit 7140f67, manifest 328fab3e.
- **Port test:** f 0/203,776 mismatches, calibration 0.
- **Jobs:** f_M training 807066/807068/807070 (IG evaluation 807067/807069/807071); calib_M training 807072/807074/807076 (IG evaluation 807073/807075/807077).
- **Follow-up:** `mj_auto.sh` waits for training to finish, then runs 18 zero-shot evaluations under M and writes `MJ_AUTO.DONE` or `MJ_AUTO.FAILED`.
