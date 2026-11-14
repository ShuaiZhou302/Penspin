# Takeover Log

## 2026-10-17 08:50 EDT (approx.) — Codex final handover live check

- The user asked to hand the course project over to Claude Code and explicitly asked to pause Codex's scheduled checks. automation penspin-allegro confirmed PAUSED; no running jobs were cancelled.
- 728018 RUNNING, 1212/2401 rounds, 29,786,112 labels; 728019 waiting on its dependency as expected.
- 729432_7 COMPLETED 5:04: WuJi with new joint solref .01, first seed43 dynamic run, max depth .971796mm, joint 0, finite/overflow normal. Independent full native check / the other two seeds / static / newcache / 8192 still pending.
- 728011 still FAILED, 728012 already CANCELLED. No long training run was restarted, and no other experiments were added during the handover.
- The reference copies of the real scripts in this package were checked against the remote SHAs; local PowerShell script syntax was checked.

## Template for subsequent Claude entries

Each entry records only: time, job ID, source directory and SHA, changes/budget, actual results, whether the four gates passed, next step, and whether user action is needed. If the outcome of a new job submission is uncertain, check the queue first; do not resubmit. For final results, prefer keeping the source JSON path and SHA rather than only numbers from chat.

## 2026-10-17 21:24 CST (09:24 EDT) — Claude takeover live check

- 728018 RUNNING 15:48, 1256/2401, 30,867,456 labels, ~45.7s/round, ~14.5h estimated remaining. 728019_[0-2] PENDING Dependency. Not resubmitted.
- 729432_7 verified: source CODE_SHA256SUMS 358 entries OK; JOINT_SOLREF_APPLIED 20×[.01,1]; rot 81.773, Dur 9.489s, custom full20+heading≥pi 2/64 (not the paper SR); first-episode penetration max .971796mm (0/64>1mm, small margin; penetration counted only for the first episode), joint 0, finite, EPA/other/initial/reset 0. Passed the measured gates.
- Submitted **729874_[6,8]** (seed42/44, allegro_cylinder, audit.sbatch from the same source directory, one bad GPU node excluded). Next: two-seed criteria → static 110 cache.

## 2026-10-17 ~21:45 CST — WuJi three-seed dynamic check passed → static/native submitted

- 729874_6/8 COMPLETED. seed42: rot 84.853, Dur 9.690s, custom 1/64; penetration .971796mm (identical to seed43, suspected to be the same cached initial contact state; to be checked in the native run). seed44: rot 76.673, Dur 8.823s, custom 2/64; penetration .936315mm. All three seeds: joint 0, finite, EPA/other/endpoint/reset/initial all 0; logs confirm JOINT_SOLREF/PRIVATE_WARP/EPA64.
- New directories (copied from the V2 manifest, 362 files each, SOURCE_V2_SHA256SUMS kept):
  - `v7_joint_solref001_static_20261017` CODE_SHA256SUMS sha 9e796c02…; bound.py/allcache_static.py from the old static template (sha 157e19b8…/be20da70…), new static_launch.py (private warp+EPA24→64 assertion). Submitted **729890_[0,1]** (seed43/44).
  - `v7_joint_solref001_native_20261017` CODE_SHA256SUMS sha 3cf0a77e…; native_peaks.py only changes the source path. Submitted **729891_0**.
- Next: static two-seed criteria + new intersection; native 192 cases over1mm/error; then prepare a new 8192 guard directory.

## 2026-10-17 ~22:40 CST — Static/native passed, new intersection, 8192 guard submitted

- Static 729890: seed43 85/110, seed44 79/110, all criteria passed; both seeds all-step penetration max .971796mm/0 over 1mm, joint 0, finite, overflow 0.
- native 729891: 192 cases native_over1mm 0, max native .971829mm, max |Warp−native| .000874mm; per-case warp matches the new source. 153/192 are identical to the old native run, because the deepest state is the cached initial pose (seed42/43 cache104 .9718mm, seed44 cache55 .9363mm).
- Summary `v7_joint_solref001_audit_20261017/summary.json` (passed_measured_physics=true); rot mean 81.100±4.131, Dur 9.334±.454s (± is the sample std over 3 evaluation seeds, not training-seed variance).
- New intersection 68: phase [12,20,6,14,8,8], −32 +92 relative to the old 68 (not a reuse of the old intersection).
- Guard directory `v7_joint_solref001_guard_20261017` (364 files, CODE_SHA256SUMS 694e2ecc…, filtered cache 9b26d624…, CACHE_PROVENANCE_CURRENT.json). Submitted **729969_0** (8192×12×10, save/load+10 closed-loop frames).
- Next: guard result; once everything passes, report to the user, then decide whether to start a new-tag long training run (not started automatically).

## 2026-10-17 ~22:50 CST — 8192 guard passed

- 729969_0 COMPLETED 4:34. JOINT_SOLREF_APPLIED confirmed; PAPER_INTERFACE_OK; 10 updates 983,040 transitions; SMOKE_SAVE_LOAD_CLOSED_LOOP_OK.
- PHYSICS_GUARD passed: 10,400 physics steps / 85,196,800 worldsteps / 260 forward; max penetration .971796mm (same cache104 initial state), joint 0, no finite/overflow failures.
- All gates (3-seed dynamic, 110 static, native, new intersection, 8192 guard) passed. **Long training not submitted; waiting for user confirmation.**

## 2026-10-17 ~23:00 CST — User approved; WuJi new-tag long training run submitted

- Directory `v7_joint_solref001_teacher_20261017` (copied from the guard directory manifest + guarded_launch(SMOKE=0,5087)/final_eval from 728011; 368 files, CODE_SHA256SUMS 20f46ee8…, TRAINING_PROTOCOL.json 15b5f91f…, cache 9b26d624…).
- Tag: joint_solref001_new68; 500,072,448 steps, seed42 random initialization, guard throughout. Relative to 728011, only joint limit solref .02→.01 + new 68 cache (−32 +92). Not a pure pen-geometry control.
- Training **730064_[2]** (40h walltime, ~25h expected, estimated from 728011 at ~17.9s/update); final evaluation **730065_[6-8]** afterok:730064, TRAIN_ID=730064, seed42/43/44×64. Do not resubmit.
- Monitoring: progress.jsonl / logs/730064_2.out; on failure see outputs/teacher_730064_2/physics_guard_failure*.

## 2026-10-17 ~23:30 CST — 730064 started

- 730064_2 RUNNING: iteration 70, 6,881,280 steps, elapsed 1308s (~19s/update, ~27h total expected), reward 6.28, Dur 1.84s; guard not triggered.

## 2026-10-18 ~05:00 CST — Live check

- 730064_2 RUNNING 5:31, iteration 1040/5087, 102,236,160 steps (already past iteration 660, where 728011 failed), reward 32.30, Dur 6.11s, guard not triggered; ~19s/update, ~21h remaining.
- 728018 RUNNING 22:47, 1805/2401, 44,359,680 labels, ~7.5h remaining. 728019/730065 waiting on dependencies.

## 2026-10-18 21:55 CST (09:55 EDT) — Allegro final results

- 728018 COMPLETED 30:17:08, 2401/2401, 59,006,976 labels, full budget. 728019_[0-2] COMPLETED.
- Summary `<PROJECT_ROOT>/penspin-officialtips-comparison-20261016/final_comparison_728019.json` (sha 9f466856…, local copy in results/). Field mapping verified bit-for-bit against the strict version's seed42 values; the strict version recomputed as 16.108876±1.107375, matching the record.
- Official-tip pureDAgger: rot 16.561±3.415, Dur 2.536±.436s, custom 0/192, timeout 0. Strict version: 16.109±1.107, 2.756±.207s, 0/192. Difference rot +0.45 (seed 42/43/44: +0.86/+2.71/−2.22), Dur −0.22s; endpoint terminations 36–41 vs 22–24. ± is the sample std over evaluation seeds, single training seed.
- Conclusion: restoring the official fingertip branch gives no distinguishable closed-loop improvement at the same label budget.
- WuJi 730064: 4520/5087, 444,334,080 steps, no guard failures, ~3h remaining.

## 2026-10-19 01:50 CST (10/18 13:50 EDT) — WuJi new-tag long training final results

- 730064_2 COMPLETED 25:56:42, 5087/5087, 500,072,448 steps. PHYSICS_GUARD passed throughout training: 4,883,520 physics steps / 40,005,795,840 worldsteps / 122,088 forward; max penetration .971796mm, joint 0. guard_palm_peak_sequence.json is a peak-window record (world6283 .870mm, below the gate), not a failure.
- 730065_[6-8] COMPLETED (final.pth, new68 cache): rot 86.181±1.857, Dur 9.356±.272s, full20 0/192, custom 0/192 (not the paper SR). All three seeds pass the dynamic physics gates: first-episode penetration max .9718mm, 0 over 1mm, joint 0, finite, overflow 0.
- Summary `v7_joint_solref001_teacher_20261017/FINAL_SUMMARY_730065.json` (sha 495d147b…, local results/).
- Comparison basis: frozen v6 teacher under the same new physics (110 cache) 81.10±4.13/9.33s, custom 3/192; v6 full run 713831 allegro_cylinder 97.98±6.11/10.59s/32/192 is physically invalid (penetration up to 16.35mm). Cache distributions differ; all are single training seed.

## 2026-10-20 ~12:50 EDT — Investigation of WuJi early termination (read-only, no new jobs)

- All 192 episodes over the three seeds of 730065 ended on the height bounds: point_low 96, center_low 66, both at once 30; no unstable/point_high/timeout. Median termination time 7.9s (p10 5.1s, p90 15.2s).
- Steady state: 2 fingertip contacts, pen-axis tilt ~5°, center 2–4mm above the grasp height, steady rotation speed (rotation ~9.2/s, about 5.8s per half turn; median 1.41 half turns completed before failure).
- Failure is a fast loss of grasp: in the ~5 frames (0.25s) before failure, fingertip contacts go 2→1→0, tilt goes 6°→13°→40°, and the center drops to −9.9mm within 1 frame. point_low failures are mainly pen tilt; center_low failures are the whole pen falling.
- The absolute heading (mod 180°) at grasp-loss onset is moderately concentrated (Rayleigh R=.223, p≈7e-5, peak at 45–105°), suggesting a link to a specific finger-gaiting phase; heading relative to the start shows no concentration (p=.62).
- The frozen v6 teacher under the same physics shows the same pattern (point_low 80/center_low 72/both 33, 5 timeout).

## 2026-10-20 13:05 EDT — Failure episode videos

- Render job **742879** COMPLETED 2:22 (`v7_joint_solref001_render_20261020`, kinematic replay of the saved 20Hz qpos/mocap + mj_forward, no simulation; teacher directory read-only, no bytecode written). Four seed42 clips: point_low world14, center_low world33, both world27, longest world41. Local `videos_730065/`, SHAs match the remote.
- Visual inspection: point_low failures are the pen pitching over around one fingertip (tilt 0→50°, center barely drops).

## 2026-10-22 — Plan to add object pose to the student (read-only check, no jobs)

- Paper check: §3.2 and Table 1 state that proprioceptive DAgger does not converge in simulation and drops the pen within the first few steps; the student is BC-pretrained on teacher rollouts (~50M) and fine-tuned with real-robot replay. Table 2 Only Pretrain, real robot A 15.15%.
- AdaDex (arXiv 2610.28628): the policy directly uses proprioception + object pose + BPS shape, no distillation; code not released.
- Trainer check: the dagger_train.py buffer stores only obs/proprio/target; env already provides obj_ends (both endpoints relative to the hand root, official ±pen_radius uniform noise, with history); the official "ends" branch in models.py (end_feat_extractor MLP[6,6,6] + max over the two ends for symmetry) exists only in the non-proprio branch and always together with fingertips. 713476 has iter_400 (9,830,400 labels), usable as a same-budget control; it has not been evaluated yet.

## 2026-10-22 — Student object endpoint experiment: new directory + smoke

- User approved the plan. New directory `<PROJECT_ROOT>/penspin-student-objpose-dagger-20261022` (copied by manifest from the strict pureDAgger source, STRICT_SOURCE_SHA256SUMS kept; local git 135803b; SOURCE_SHA256SUMS 46 entries).
- Modes: proprio (same as the strict version, 513,993 parameters, can strictly load 713476 iter_400); A proprio_ends_gt (simulator endpoints, official ±radius noise, 523,317); B proprio_ends_est (endpoints estimated from proprioception + auxiliary L1 in cm, weight 1.0, 526,599; input-perturbation assertion that it does not read obj_ends). CPU tests passed.
- Smoke: 749893 (A, 2 rounds), 749894 (B, 2 rounds); evaluation-chain equivalence check 749895 (proprio mode evaluating 713476 final seed42, should reproduce 15.434128).

## 2026-10-23 — Smoke failure diagnosis and resubmission

- 749893/749894/749895 all FAILED 1:01, all on one bad GPU node, `Device count 0` → `CUDA unknown error` (node GPU not visible); the code never ran. Resubmitted with that node also excluded: smoke A **754009**, B **754013**, equivalence check **754014**.
- 754014 equivalence check COMPLETED: the new evaluation chain in proprio mode reproduces 713476 final seed42 rotation 15.434127807617188 / Dur 2.6812500953674316, bit-for-bit identical.
- 754013 B smoke COMPLETED: OBJ_ENDS_DEPENDENCY delta 0 (does not read simulator endpoints); pose_l1 4.54→2.81cm, rollout 13.98→3.20cm; frozen-teacher check passed.
- 754009 A smoke hit another bad GPU node (Device count 0) → resubmitted **754030** (excluding the bad nodes). B evaluation-chain smoke **754031**.
- Submitted the B full 400-round run **754033** (12h walltime) and the three-seed evaluation of baseline 713476 iter_400 **754034_[0-2]**. The A full run waits for its smoke to pass.
- 754030 A smoke COMPLETED: OBJ_ENDS_DEPENDENCY delta 7.5e-4>0 (does use endpoints). 754031 B evaluation-chain smoke COMPLETED (EVAL_OBJ_ENDS_DEPENDENCY 0).
- Submitted the A full 400-round run **754045**. B 754033 RUNNING.
- Same-budget baseline 754034 (713476 iter_400, 9,830,400 labels): rot 13.633±1.758 (13.036/15.612/12.252), Dur 2.465±.301s, custom 0/192, timeout 0.

## 2026-10-24 03:56 CST — A/B midway (training metrics, not closed-loop conclusions)

- B 754033 iter147: loss 2.847, rollout L1 2.957, training episode length 48.5 steps, pose 2.20cm. A 754045 iter137: loss 2.248, rollout 2.292, training episode length 69.8 steps. Strict version at the same rounds: loss 2.85/2.83, episode length 53.6/55.6. About 45s/round, ~3.2h remaining.

## 2026-10-24 07:15 CST — B training done, evaluation submitted

- B 754033 COMPLETED 4:53:47 (400 rounds, frozen-teacher check passed) → evaluation **755325_[0-2]**.
- A 754045 iter395, loss 2.017, training episode length 88.8 steps (longer episodes make each round slower); its evaluation is submitted automatically on completion.

## 2026-10-24 — Student object endpoint results (400 rounds / 9,830,400 labels, single training seed, 3 evaluation seeds×64)

- A 754045 COMPLETED, evaluation 755336; B evaluation 755325. Evaluation-side dependency assertions: B delta 0 (does not read endpoints), A delta .44–.49.
- Strict iter_400: rot 13.633±1.758, Dur 2.465±.301s, ever180 7/192, custom 0.
- **A proprio_ends_gt: rot 33.915±2.458, Dur 4.564±.352s, ever180 69/192, timeout 1, custom 0** (per seed 31.19/35.95/34.60; all three seeds are above every baseline seed).
- B proprio_ends_est: rot 14.525±1.954, Dur 2.519±.256s, ever180 5, custom 0 — no distinguishable difference from the baseline (the endpoint estimate error plateaued at ~2.2cm during training).
- Summary `penspin-student-objpose-dagger-20261022/COMPARISON_400.json` (sha d3305279…, local results/objpose_comparison_400.json).

## 2026-10-24 — Endpoint noise sensitivity (user approved)

- Code: ENDS_NOISE_CM (proprio_ends_gt only) adds i.i.d. Gaussian noise to the 18-dim endpoint history, resampled every control step / every minibatch / every evaluation step, σ=cm/100/√(2/π) so that the per-coordinate mean absolute error = cm (same measure as B's pose_l1_cm); separate generators (training 24001, evaluation 24002), global RNG untouched; default 0 is identical to A.
- Commits: 2f187ee (docs + patch file; the assembly script missed the line that applies it, so the code was unchanged) → b2d026c (patch actually applied). Old manifest saved as SOURCE_SHA256SUMS_v1_135803b; new SOURCE_SHA256SUMS 48 entries. CPU check of noise 1/2/3cm measured .999/1.999/2.998.
- Smoke 2cm: **755389**. Limitation: i.i.d. unbiased noise ≠ the estimator's time-correlated/biased error.
- Smoke 755389 COMPLETED: ENDS_NOISE measured 2.003cm, dependency assertion >0, frozen-teacher check passed.
- Full runs (400 rounds each, b2d026c): 1cm training **755392**→evaluation **755393**; 2cm **755394**→**755395**; 3cm **755396**→**755397** (afterok dependency; evaluation uses the same ENDS_NOISE_CM). Do not resubmit.

## 2026-10-24 17:45 CST — Endpoint noise sensitivity results (400 rounds, single training seed, 3 evaluation seeds×64)

- All training runs include the frozen-teacher check, 400 rounds/9,830,400 labels; evaluation logs show EVAL_ENDS_NOISE_CM matching training.
- rot / Dur / ever180: baseline 13.63±1.76 / 2.47 / 7; B 14.52±1.95 / 2.52 / 5; A0 33.91±2.46 / 4.56 / 69; **A1cm 32.67±3.74 / 4.25 / 64; A2cm 33.13±4.44 / 4.48 / 61; A3cm 27.80±4.11 / 3.84 / 55**.
- B estimator error is 2.19cm, while a "constant mean prediction" on B's own visited states gives 3.03cm (median 2.99cm) → the estimator is only ~27% better than a constant.
- Conclusion: A is almost insensitive to ≤2cm unbiased noise; it drops clearly only at 3cm and is still about 2× the baseline. B's failure is not about error magnitude: proprioception contains very little extractable pose information (the estimate is close to a constant).
- Summary `COMPARISON_NOISE_400.json` (sha 3b62b87c…, local results/objpose_noise_comparison_400.json).

## 2026-10-24 — B+tactile (user approved)

- New mode proprio_tactile_ends_est: the estimator additionally reads the env's binary tactile history (30×32, official threshold/latency/dropout sensor model), separate TemporalTransformer → concatenated with proprio features → MLP[64,18]; tactile enters the policy only through the 18-dim endpoint estimate (policy input 154, same as B). 804,743 parameters (B 526,599; capacity confound).
- Commit f63f1d7; manifest 50 entries (previous version saved as SOURCE_SHA256SUMS_v2_b2d026c). CPU tests: uses tactile, does not read simulator endpoints, other modes unaffected by tactile.
- Smoke **758749** (2 rounds).
- Smoke 758749 COMPLETED: does not read simulator endpoints (delta 0), tactile is used (delta 7e-4); tactile being all 0 at the first step is due to reset zeroing; reservoir check: tactile is 0/1, 94.7% of samples have contact, 9.6% of sensors active on average. pose 5.28→2.77cm.
- Full 400-round run **758850** → evaluation **758851_[0-2]** (afterok). Do not resubmit.

## 2026-10-26 10:00 CST — B+tactile results

- 758850 COMPLETED 5:18:02 (frozen-teacher check passed); the 758851 evaluation asserts no endpoint reads on all three seeds.
- **B+tactile: rot 43.87±3.10 (41.4/47.3/42.9), Dur 5.56±.31s, ever180 79/192, timeout 4, custom 0**; end-of-training loss 1.966, training episode 102 steps, pose 2.07cm (B 2.19cm, constant prediction ~3.0cm).
- Comparison: baseline 13.63; B 14.52; A (true endpoints) 33.91. B+tactile is above A, but the endpoint estimate is barely more accurate → the gain does not come from a more accurate pose; the 18-dim "endpoint" channel is shaped by BC gradients and may carry non-pose information such as contact timing; there is also the confound of a larger estimator (about 805k vs 527k parameters). Single training seed.
- Summary `COMPARISON_ALL_400.json` (sha 349e1c69…, local results/objpose_comparison_all_400.json).

## 2026-10-26 — Two controls (user approved)

- proprio_tactile_ends_est_sg (estimate detached, estimator trained only by the pose loss; 804,743 parameters); proprio_extra_ends_est (B + an extra proprio transformer, equal capacity 804,743, no tactile). CPU tests: sg has BC→estimator gradient 0 and pose gradient >0; extra does not read tactile; other modes unchanged.
- Commit 5133171; manifest 52 entries (previous version SOURCE_SHA256SUMS_v3_f63f1d7). Smoke: sg **765787**, extra **765789**.
- Smoke 765787 (sg): SG_GRADIENT_CHECK bc→estimator 0, pose 1.9e4; 765789 (extra) normal; both passed the frozen-teacher check.
- Full 400-round runs: sg **765801**→evaluation **765802**; extra **765803**→evaluation **765804** (afterok). Do not resubmit.

## 2026-10-28 03:35 CST — Results of the two controls

- 765801/765803 COMPLETED (frozen-teacher check passed); evaluations 765802/765804 both assert no endpoint reads.
- **sg (tactile only through a pure pose estimate): rot 24.91±3.77, Dur 3.57±.43s, ever180 39, pose 2.08cm**.
- **Equal capacity, no tactile: rot 14.56±2.94, Dur 2.55±.37s, ever180 14, pose 2.18cm** — same as B, which rules out capacity.
- Decomposition (400 rounds, single training seed): B 14.5 → +tactile as pose estimate (sg) 24.9 → +BC-shaped endpoint channel (non-sg) 43.9; true endpoints A 33.9. Tactile provides both pose information and information beyond pose (contact timing etc.), and the latter contributes more. The mean estimate error is almost unchanged (2.07–2.19cm); mean error does not capture the information that is useful for control.
- Summary `COMPARISON_CONTROLS_400.json` (sha 45c07f9a…, local results/objpose_comparison_controls_400.json).

## 2026-10-28 — Training seed replication (user approved)

- 61d301b: TRAIN_SEED (default 0 = all previous runs) sets the train.py seed and the reservoir sampling seed 13001+seed; previous manifest SOURCE_SHA256SUMS_v4_5133171.
- Smoke **772986** (B+tactile seed1, 2 rounds); full runs all afterok:772986: A seed1 **772987**→evaluation **772988**; A seed2 **772989**→**772990**; B+tactile seed1 **772991**→**772992**; the first submission of B+tactile seed2 failed to submit (no job ID, no duplicate), then resubmitted **772994**→**772995**. Do not resubmit.
- Smoke 772986 COMPLETED: seed1 initialization differs from seed0 (first-round loss 3.822 vs 3.681); dependency assertions and frozen teacher normal. A s1/s2 running, B+tactile s1/s2 queued.
- All COMPLETED. Results `results/objpose_comparison_seeds_400.json` (sha ff8d98be…, remote COMPARISON_SEEDS_400.json; script seed_compare.py checks config seed, 400 rounds/9830400 labels, frozen teacher, noise-free evaluation). Rotation reward (± = std over 3 training seeds): A **35.30±1.61** (33.91/37.07/34.91), B+tactile **42.87±2.23** (43.87/44.42/40.31); Duration 4.72 vs 5.51 s. B+tactile > A > baseline 13.63 holds on every seed. The baseline and controls still have only a single training seed. The custom metric is not the paper SR.

## 2026-10-28 — Allegro student gap diagnosis (user approved "run both")

- Failure breakdown (existing evaluations): rotation per step, student B+tactile .0255 vs teacher 705613 .0283 rad/step; the gap is mainly in survival time: the teacher lasts the full 20 s in 59% of episodes, median 399 steps; B+tactile 1%, median 92 steps.
- e8a0db9: added `proprio_priv_gt` (upper-bound diagnostic, not deployable): the student additionally reads the teacher's priv_info (61-dim) and point cloud (100 points), using the frozen teacher normalizer and a newly initialized encoder; use_ends now checks whether the mode name contains 'ends' (existing modes behave the same). manifest 04862fe7…, previous version SOURCE_SHA256SUMS_v5_61d301b. CPU tests: parameter counts of existing modes unchanged, priv mode 586,513 parameters.
- Smoke: 774485 (priv, 2 rounds, PRIV_DEPENDENCY passed), 774486 (priv evaluation seed42, EVAL_PRIV_DEPENDENCY passed), 774487 (resumed from the round-400 state of 758850 for 2 rounds, RESUME_VERIFIED, loss 1.983), all COMPLETED.
- Full jobs, do not resubmit: priv 400 rounds **774499**, evaluation **774500**; B+tactile seed0 resumed from round 400 of 758850 to round 1200 **774501**, evaluations at round 500 **774502**, round 1000 **774503**, round 1200 **774504**. All evaluations are afterok on their training jobs.
- 774499/774500 done. The priv upper-bound student's rotation reward is **122.11±16.21** (± is the std over 3 evaluation seeds, single training seed), Duration 13.49 s, timeout 109/192. Under the same protocol teacher 705613 gets 123.53±7.46, Duration 13.70 s, timeout 114/192. Final training loss 0.286, mean training episode length 304 steps. The same DAgger trainer reaches teacher level in 400 rounds → the bottleneck is information, not the learning pipeline. Results `results/objpose_comparison_priv_400.json` (sha 872c09f1…). The 774501 resumed training is still running.

## 2026-10-28 — SuperDex step 0 (course proposal; user: no teammates, build from scratch)

- Ablations (fa99bdc, manifest 3815f3f9…, previous version _v6_e8a0db9): smoke 776537 (no_pc) / 776538 evaluation, 776539 (no_objpose) / 776540 evaluation, still queued. The 6 full ablations are submitted only after these pass.
- New directory `<PROJECT_ROOT>/penspin-superdex-port-20261028`. Installed from the superdex 1.0.1 manylinux_2_28 wheel (glibc 2.28). Created `.venv` with uv (Python 3.12.4) and installed superdex, physics, robotics, lab, all version 1.0.1.
- PenSpin's Allegro URDF (round_tip/allegro_hand_right_fsr_round_dense.urdf) loads: 16 DoF, the prefab has 55 links (the URDF has 67; the difference is still to be checked); 7 links have invalid inertia tensors, with warnings. The runtime URDF loader ignores primitive shapes such as box/cylinder, and this hand's collision geometry is mostly boxes, including the FSR tactile pads, so they must be converted to closed meshes.
- All the needed interfaces exist:
  - State: Scene.capture_state/restore_state, actor.set_articulated_pose_from_joints/set_root_transform/set_velocity;
  - Control: ControllerBasicJscPd (kp/kd/saturation), or set_articulated_target_pose;
  - Contact: get_contact_points_world/get_contact_force_world/get_contact_force_from_actor_world (ContactPoint carries force and normal);
  - ContactParams: Coulomb friction, penalty compliance; usable for friction/compliance sweeps;
  - No ready-made tactile sensor class, but register_python_sensor exists.
- Without contact, 120 steps take about 0.01 s.

## 2026-10-29 — SuperDex step 1: port and open-loop replay

- `tools/convert_urdf.py`: in the PenSpin URDF, the 51 box collisions are converted to closed meshes. Mesh files must not have dots in their stem and must use absolute paths, or the loader silently skips them. The two extra base_link boxes are split into fixed child links. Each of the 32 FSR pads gets mass computed at 1000 kg/m³. All meshes are watertight.
- `tools/sdx_env.py`:
  - Reproduces IG's 200 Hz torque PD: p3, d0.1, ±0.7 clip; D is 0 on the first substep of each control step.
  - Joint friction .01, armature .001, IG calibrated joint limits, hand pose, no gravity on the hand.
  - Pen is a cylinder r .04s, l .4s.
  - sdp.Quaternion is in (x,y,z,w) order.
  - Base-welded links have no contact sample points, so per-link contact force is summed from the pen-side contact points.
- **Isaac Gym quirk**: `link_13.0_fsr` does not exist in the URDF. IG returns -1 for it, and `contacts[:, -1]` reads the object's net contact force. So the 12th tactile channel is actually "is the pen being touched", which also affects the earlier B+tactile results. The recorder asserts this, and SuperDex reproduces it with the pen's total contact force.
- Isaac copy: `penspin-superdex-port-20261028/isaac`, cloned from fa99bdc, with recording hooks added.
  - record 776747: 64 teacher episodes, no random forces, no action noise, no PD/COM/friction randomization. Mean 297 steps, tactile on 10.8%.
  - IG self-replay 776770 (same seed, recorded targets replayed open-loop): trajectories are deterministic (median pen-position difference 0.0 cm, binary tactile agreement 99.6%), and 80% of episodes reach the original length.
- SuperDex replay of all 64 (`runs/replay_fid1_all64.*`, 128 s):
  - q RMSE median 0.008 rad; pen-position error at 1 s, median 0.69 cm.
  - **The pen survives a median of 28.5 steps (1.4 s) vs 398 in IG.** Only 11% of episodes reach the IG length.
  - Tactile on-rate IG 11.7% vs SDX 3.1%; per-channel F1 0.40.
- Perturbed IG self-replays (p+1% 776795, p+5% 776796) are running, to measure how sensitive open-loop replay is to small dynamics differences.
- Ablation smokes 776537–776540 are still queued, and 774501 is running (round 565 → 1200).
- First perturbed self-replay attempt via a hydra pgain override failed: the reset warmup step changed with the gain, and envs that terminated during warmup resampled their grasp. Fixed by scaling `env.p_gain` *after* the identical reset (strict 1e-6 assert on the initial state). 776849 (×1.01) and 776850 (×1.05): IG open-loop survival drops from a median of 398 steps to **97.5 / 82.5** (17% / 20% reach the original length), with pen offset at t=10 of 0.06 / 0.05 cm. SDX: 29.5 steps, 0.54 cm. → Open-loop replay is extremely fragile, so survival time can't measure sim similarity; SDX's early divergence is still about 10× that of a 1% PD perturbation.
- 776854 (fid2, trajectories identical to fid1) records the object's actual rigid-body properties. **IG pen inertia is isotropic, 1.4–3e-7 kg·m²**: transverse inertia is about 1/85 of a solid cylinder's, and axial about 1/5. (The URDF hard-codes 1e-4, then scale and mass randomization shrink it; COM ≈ 0.) To be checked against the upstream repository. In SDX, switching to IG inertia (sdp.Real6, read back and confirmed) gives almost the same replay (29.5 steps, 0.53 cm), so inertia is not the main cause of the open-loop divergence.
- The divergence starts at the first control step: at t=0 (50 ms) the pen already differs by 0.32 cm (IG moves it 0.19 cm, SDX 0.32 cm), and max|dq| is 0.014 rad, then grows slowly afterwards. This fits the different contact models (PhysX rigid contact with 2 mm contact_offset vs SDX penalty/soft contact) responding differently to the initial grasp contact. No port bug found yet.

## 2026-10-28 — Teacher closed loop in SuperDex

- isaac: the recorder now also stores the teacher's raw inputs (obs96, priv61, point cloud), actions, fingertip positions and local point clouds; `export_teacher.py` → `assets/teacher_705613.npz`. record 776864 (fid3, obs noise 0): 64 episodes, mean 295 steps.
- `tools/teacher_np.py` is a numpy port of the teacher plus the IG observation pipeline. `test_teacher_port.py` results:
  - action max diff 2.4e-6;
  - obs96 0, non-tactile priv 7e-6, point cloud 6e-8, initial history frames 0;
  - tactile on-rate: raw 11.0%, after latency/dropout 9.9%.
- Note: indexing a compressed NpzFile inside a loop decompresses the whole array every time, so scripts now use dict(np.load(...)).
- `tools/closed_loop.py`, 64 episodes from the same initial states, no obs noise, friction 1.0:
  - **IG teacher**: rotation reward 139.9, Duration 14.7 s, median 398 steps.
  - **SuperDex teacher**: 42.2 (physical inertia) / 41.2 (IG inertia), Duration 5.2 / 5.0 s, median 60 steps; 5 / 4 of 64 episodes reach 20 s.
  - Rotation per step is close (SDX .030 vs IG .033 rad/step). As with the student, the gap is in how long the grasp holds; pen inertia has almost no effect.
- Friction sweep (0.5/0.75/1.5/2.0, physical inertia) is running in the background (detached).
- SDX friction sweep (teacher closed loop, physical inertia, 64 episodes). μ is pen rotation reward / Duration / episodes reaching 20 s:
  - μ=0.5: **68.6 / 7.64 s / 11**
  - μ=0.75: 61.9 / 7.08 s / 11
  - μ=1.0: 42.2 / 5.23 s / 5
  - μ=1.5: 30.3 / 4.00 s / 0
  - μ=2.0: 24.8 / 3.46 s / 0
  
  In SDX, lower friction works better for the teacher. The IG side uses μ=1.0, so SDX's effective friction looks higher than IG's; friction combination and stiction differences are worth checking.
- 774501 (resumed from round 400 of 758850 to round 1200, 29,491,200 labels; RESUME_VERIFIED, frozen teacher check passed; final loss 1.763, pose error 2.06 cm). Evaluation rotation reward (± is std over 3 evaluation seeds): **round 500 37.56±2.94 (774502), round 1000 45.35±8.10 (774503), round 1200 44.40±5.78 (774504)**, vs round 400 43.87±3.10. Tripling the labels gives essentially no gain → more evidence that the bottleneck is information.
- Ablation smoke 776537/776538 (no_pc) passed: PRIV_DEPENDENCY and EVAL_PRIV_DEPENDENCY both confirm that the dropped group has 0 effect. 776539/776540 (no_objpose) still queued. Submitted 6 full runs with afterok:776539:776540 (fa99bdc) — **do not resubmit**:
  - objpose 776977 → 776978
  - angvel 776979 → 776980
  - phys 776981 → 776982
  - ftpos 776983 → 776984
  - contact 776985 → 776986
  - pc 776987 → 776988

## 2026-10-28 — SuperDex μ=1.0 zero-shot baselines (user approved the recommended target domain)

- Target domain is fixed: SuperDex μ=1.0, physical pen inertia, default contact parameters. μ=0.5/1.5 are only sweep points.
- Evaluation protocol: Isaac Gym seeds 42/43/44 × 64 episodes from identical initial states. Obs noise t.01/e.02; no external force, action noise, or PD/COM/friction randomization.
- isaac aba5425: recorder `RECORD_POLICY=student`. IG baselines:
  - teacher 777022: 141.0 / 127.0 / 140.9
  - proprio 713476 iter400, 777023: 15.9 / 12.9 / 13.7
  - B+tactile 758850, 777024: 40.5 / 49.1 / 42.3
- SDX venv now has torch 2.7.1 (`tools/student_policy.py` imports the frozen model code directly).
- Student port check (`test_student_port.py`):
  - action max diff 1.9e-6 / 1.7e-6;
  - initial frames, frame shift, newest target and tactile frame all 0;
  - noise std per episode .020, per step .0099.
- **Zero-shot result** (rotation reward, ± over 3 eval seeds; `results/superdex_zeroshot_mu1.json`, sha 67486be1…):

| Policy | IG | SDX μ=1.0 | SDX/IG | SDX Duration |
|---|---|---|---|---|
| Teacher | 136.3±8.1 (14.3 s) | 35.1±5.6 | 0.26 | 4.66 s |
| Proprio student | 14.2±1.5 (2.24 s) | 12.8±0.8 | 0.91 | 2.19 s |
| B+tactile student | 44.0±4.5 (5.22 s) | **6.6±0.6** | 0.15 | 1.50 s |

  → Tactile helps a lot in IG (44 vs 14), but in SDX it **falls below proprio-only** (6.6 vs 12.8). This matches the proposal's premise: a student that relies on one simulator's contact signal breaks when that signal changes. Tactile on-rate is IG 11.7% vs SDX 3.1%.

## 2026-10-29 SuperDex step 3 (paired contact data) + tactile port bug
- Recorder 3f7183c (isaac clone): stores per-step qd and object world velocities. IG records (obs noise on): pairT seeds 100-102 (777112, teacher rot reward 135.1/135.1/137.1), pairBT seed 103 (777113, B+tactile student, 43.6). SHAs in runs/RECORDS_PAIR_SHA256SUMS.
- Pairing = step-wise resync (SDX reset to the IG state before each step, one control step at the IG target), plus plain open-loop replay. Checks: SDX pen velocities are world-frame; reset is history-free (reused vs fresh env diff 0). Open-loop pairs are unusable (only 7% of steps have pen error < 1 cm); resync one-step error: joints 0.004 rad median, pen 0.19 cm, 3.7 deg.
- **BUG found in sdx_env.py (all earlier SDX tactile numbers invalid):** actors were keyed by str(ActorHandle) = Python repr (memory address), so per-link FSR forces were dropped; SDX tactile was all-zero except the pen-total channel (on-rate 3.1% = 1/32). Fixed: key by ActorHandle.value (sdx_env sha 54be0b53...). Invalid: replay contact F1/on-rate, teacher closed loop + friction sweep, B+tactile zero-shot (6.6), runs/pairs/. Unaffected: joint/pen kinematics, proprio-student zero-shot (12.8±0.8). All tactile-dependent runs rerun into runs/v2 and runs/pairs_v2.
- v2 (fixed) SDX mu=1.0 zero-shot, rot reward IG -> SDX (3 eval seeds): teacher 136.3±8.1 -> **37.5±1.6**; proprio 14.2 -> 12.8±0.8 (unchanged); B+tactile 44.0±4.5 -> **16.4±1.1** (keeps 37% vs proprio 91%, but still above proprio; the v1 "below proprio" claim was the bug). Teacher no-noise mu sweep v2: 0.5 66.9 / 0.75 56.8 / 1.0 43.7 / 1.5 32.0 / 2.0 28.3.
- pairs_v2 (dataset 548d981b): train 119 eps / 36,265 steps, val 62 / 18,393, shift (B+tactile) 62 / 6,598. Identity baseline (SDX contact := IG contact), F1 excluding the pen-total channel: 0.63 / 0.62 / 0.64. SDX under-detects: on-rate 4.6% vs IG 7.9%, half of IG contacts missing in SDX, 16% of SDX contacts not in IG; worst tip_fsr_5 pads. Saved results/superdex_v2_zeroshot_pairs.json.
- Step 4, f v1 (tools/train_f.py sha 1f519980, runs/f_v1, results 95e019e9): MLP 256x2, inputs IG log-force x3 steps + q_t, q_{t-1}, target, action; 4000 AdamW steps, no val selection. Micro F1 excl. pen channel (mean±std over 3 f seeds):
  val (held-out teacher): identity 0.621 | per-channel threshold 0.67 | f 10 eps 0.742±0.007, 25 0.786±0.004, 50 0.807±0.004, 100 **0.822±0.003**.
  shift (B+tactile student rollouts): identity 0.638 | threshold 0.68 | f 10 eps 0.577 (below identity), 25 0.658, 50 0.668, 100 **0.706±0.009**.
  Identity P/R 0.49/0.84 (IG over-reports contact); f@100 P/R 0.84/0.80 on val. Gain shrinks under policy shift -> student-state pairs (DAgger-style) are the obvious next data source for f.
- f v3 with student-rollout pairs (dataset_mu1_v3 bd05b2c1; IG records 777416 B+tactile seeds 104/105, 777417 proprio seeds 106/107/108; train_bt 122 eps/12.2k steps, train_p 119/5.4k; new held-out shift_p = proprio seed 108, 61 eps/2.9k). Micro F1 excl. pen ch, mean of 3 f seeds, val / shift(B+tactile) / shift_p(proprio):
  identity 0.621/0.638/0.627 | threshold ~0.67/0.68/0.67 | teacher-only@100 0.822/0.706/0.654 | student-only 0.737/0.755/0.731 |
  teacher+student @10 0.793/0.760/0.732, @25 0.800/0.764/0.733, @50 0.814/0.767/0.739, **@100 0.827/0.768/0.742**.
  Student pairs close most of the policy-shift gap at no cost on teacher val. Candidate f for step 5: runs/f_v3/teacher_plus_student/f_b100_s0.pt (seed 0 by convention, no val selection).
- Step 5 (f-aligned student): repo penspin-student-falign-20261029 commit 97c4472 (manifest 69b01dc3), = objpose fa99bdc + falign_patch.py: TACTILE_ALIGN_F -> student tactile = latency/dropout(f(IG log-force x3, q_t, q_{t-1}, target, clamped action)); teacher priv tactile stays raw. f = f_v3 teacher+student b100 s0 (caedb3a9). Port test vs train_f: 0 mismatches / 203,776 outputs. Smoke 777474 OK. In-env check (record 777486 with smoke ckpt, isaac clone dce4c06 same patch): student tactile on 4.3% vs raw sensed 7.4% excl. pen ch (SDX paired 4.6%). Full run 777476 (400 rounds, same settings as 758850), IG eval 777477 (with f channel), IG records evalFA (afterok) for SDX zero-shot.
- Step 5 result (777476 COMPLETED 5:03; final imitation L1 2.355 vs 1.966 for 758850 at round 400). Rot reward (3 eval seeds, ±sd):
  IG protocol eval (with f channel, 777477): **24.9±2.4** (25.1/27.3/22.4), Duration 3.64 s — vs B+tactile 44.0±4.5 (5.22 s), proprio 14.2.
  SDX mu=1.0 zero-shot (runs/v2/zs_mu1_falign_s4*, IG records 777491): **14.7±1.8** (14.6/16.6/13.0), Duration 2.44 s — vs B+tactile 16.4±1.1 (2.54 s), proprio 12.8±0.8 (2.19 s). SDX/IG 0.60 (B+tactile 0.37).
  → f-aligned student transfers with a smaller relative drop but is **not better in absolute SDX reward** than B+tactile (one training seed; difference within eval-seed spread). Main cost is in IG: f's sparser/less informative contacts make imitation harder (higher loss, IG 24.9 vs 44.0). Proposal hypothesis not supported as-is.
- Leave-one-group-out privileged-student ablation (776977-776988, objpose repo, IG protocol eval, 3 eval seeds): drop objpose 128.0±15.4, angvel 121.6±17.8, phys 127.1±11.5, ftpos 124.1±17.6, contact 125.1±17.5, pc 121.8±17.6 (Duration 13.5-14.1 s) vs full 122.1. No single group matters; all within eval-seed spread (seed 43 consistently ~20 higher).
- Step 6 baselines (repo penspin-student-tacbase-20261029 commit 3de74e1, manifest 861abc21; isaac clone 8a2c556 same patch). Student tactile channel only; teacher priv tactile raw; same DAgger settings as 758850/777476.
  Contact randomization (TACTILE_RAND=1, ranges fixed a priori): per env/episode per-channel thresh 0.05 x logU[0.5,4], miss U[0,0.5], spurious U[0,0.01], delay {0,1,2}.
  Threshold/delay calibration (TACTILE_CALIB, tools/calib_thresh_delay.py, runs/calib/calib_b100.npz c99520ca): per-channel tau fit on the same rows as f (teacher 100 eps + student pairs), delay chosen by val F1: k=2 (val F1 0.675 vs k=0 0.673, noise-level; k=2 is worse on shift 0.663 vs 0.681 and shift_p 0.635 vs 0.670 — followed the val rule). Unit test vs offline reference: 0 mismatches.
  Smokes 778399/778400 OK. Full: RAND train 778931 / eval 778932 / record 778933 (evalTR); CALIB 778935 / 778937 / 778938 (evalTC).
- SDX friction sweep, zero-shot (runs/v2/zs_mu{0.5,1.5}_*, jobs_musweep.txt; closed_loop 1fec9e64, sdx_env 54be0b53). Rot reward, 3 eval seeds; mu=0.5 / 1.0 / 1.5:
  teacher 63.2±5.9 / 37.5±1.6 / 31.7±3.6 (IG 136.3) | proprio 10.0±0.8 / 12.8±0.8 / 13.9±0.8 (IG 14.2) |
  B+tactile 12.9±1.3 / 16.4±1.1 / 16.7±1.1 (IG 44.0) | f-aligned 14.7±0.8 / 14.7±1.8 / 14.4±0.4 (IG 24.4).
  Durations (s): proprio 1.74/2.19/2.35, B+tactile 1.94/2.54/2.67, f 2.18/2.44/2.46, teacher 7.13/4.79/4.21.
  → f-aligned is flat across mu (f fit only on mu=1 pairs) and beats B+tactile only at mu=0.5; B+tactile better at 1.0/1.5. 1 training seed each.
- runs/v2/step6_auto.sh (detached): waits for the step-6 chains, then SDX zero-shot at mu 1.0/0.5/1.5 for trand/tcalib (18 runs). Writes step6_auto.DONE or step6_auto.FAILED; no resubmission.
- Step 6 done 2026-10-31 04:49 CST (results/superdex_step6_baselines.json). All jobs COMPLETED, step6_auto.DONE, 18/18 JSON, .err empty. 1 training seed each, ± = 3 eval seeds.
  IG protocol rot: rand 25.7±3.1 (3.67 s), calib 31.4±2.0 (4.23 s); f 24.9±2.4, B+tactile 44.0±4.5.
  SDX mu=0.5/1.0/1.5: rand 14.5±1.3 / 15.8±1.2 / 16.9±2.1 (2.13/2.50/2.77 s); calib 13.0±0.7 / 16.0±0.7 / 17.3±0.8 (1.98/2.49/2.73 s).
  → f-aligned does not beat the simple baselines: below rand/calib/B+tactile at mu=1.0 and 1.5, ties rand at mu=0.5.
- Mistake in step-6 records: 778933/778938 submitted without RECORD_POLICY=student (default teacher), so they hold teacher rollouts; ig_rot_reward_mean in the trand/tcalib JSONs is teacher data, do not use it. SDX numbers are valid: closed_loop takes only the initial state; rerun from record evalFA at mu=1 gave identical means to 3 decimals (tcalib s42 64/64 episodes identical; trand/tcalib s43,s44 at most 1/64 differs, episodes with IG length <=1). Not re-recorded; needed only if these rollouts are used as paired data.
- f audit 2026-10-31 (results/superdex_penlen_audit.json; dir penspin-superdex-penlen-20261031, old dirs untouched). SuperDex bug found: all pens were 0.40 x scale while IG uses 0.40/0.44/0.48/0.52 x scale, and end-point termination used 0.40 x scale instead of the IG scalar 0.52 x last env scale. All earlier SDX numbers and pairs are affected (0.40 pens' pair labels are identical).
  SDX mu=1 fixed vs old: teacher 49.2±3.7 vs 37.5, proprio 11.7±0.9 vs 12.8, B+tactile 17.8±2.1 vs 16.4, rand 17.9±1.7 vs 15.8, calib 17.4±1.9 vs 16.0, f 14.5±1.4 vs 14.7. mu=0.5/1.5 not rerun.
  f on correct labels: val F1 0.790 (0.826 on old labels; 0.738 on 0.52 pens), own rollouts 0.722 (calib 0.632, identity 0.625). SDX on-rate 4.8% vs IG 8.1% is real, not an artifact of the short pen. Pen estimate error: SDX about 0.8 cm for all four tactile students; IG f 0.8, B+tactile 0.5.
  → f and calib must be refit on correct pairs and their students retrained before f is judged; current f student is not a valid test.
- Rebuild 2026-10-31 (user approved). All pairs re-collected with the IG pen length (penspin-superdex-penlen-20261031/runs/pairs_v4, dataset be3e4d96). f v4 (f_b100_s0 87bc57a1): val/shift/shift_p F1 0.826/0.775/0.737, seeds 1,2 within 0.005. Calib v4 (4a8fcee4): k=1, val 0.687. Unit tests 0 mismatches.
  Repo penspin-student-penlen-20261031 commit 1923a67, manifest 83df3727; environment code unchanged from 3de74e1, only f/calib files and sbatch paths.
  Jobs: f train 783805 / eval 783807 / record 783809 (evalFB); calib 783806 / 783808 / 783810 (evalTD).
  step7_auto.sh (detached): waits for those jobs, checks policy=student in the records, runs 18 SDX zero-shots (fv4, cv4; mu 1.0/0.5/1.5). Writes step7_auto.DONE or step7_auto.FAILED; no resubmission.
  SDX with the pen fix, mu=0.5 / 1.0 / 1.5 (Duration s): teacher 76.8±0.8 / 49.2±3.7 / 31.3±3.7 (8.55/6.10/4.27); proprio 10.6±1.6 / 11.7±0.9 / 13.2±1.1 (1.82/2.08/2.28); B+tactile 14.5±1.5 / 17.8±2.1 / 17.4±1.0 (2.13/2.74/2.75); rand 12.8±0.7 / 17.9±1.7 / 17.2±0.6 (1.98/2.76/2.84). Replaces the earlier sweep.
- Step 7 done 2026-11-01 14:22 CST (results/superdex_step7_fixed.json). 783805-783810 COMPLETED, records policy=student, .err empty, 18/18 JSON. 1 training seed each, ± = 3 eval seeds.
  IG protocol rot: f v4 25.4±4.1 (3.66 s), calib v4 31.2±3.8 (4.13 s). Final loss L1: f v4 2.335, calib v4 2.125.
  SDX mu=0.5/1.0/1.5: f v4 13.3±1.3 / 15.1±1.5 / 15.9±0.8 (2.09/2.46/2.75 s); calib v4 14.1±0.8 / 16.8±0.2 / 17.3±1.5 (2.00/2.41/2.57 s).
  → With correct labels f still does not beat the simple baselines: below B+tactile, rand and calib on all 3 eval seeds at mu=1.0 and 1.5; above proprio at every mu.
- Training seeds 1,2 submitted 2026-11-01 23:23 CST from repo penspin-student-penlen-20261031 (1923a67), TRAIN_SEED=1/2, f model stays f_v4 s0. train/eval: f 789400/789401, 789406/789407; calib 789402/789403, 789408/789409; rand 789404/789405, 789410/789412. No record jobs: initial states are identical across records, so each family reuses its seed-0 record. B+tactile seeds 1,2 are 772991, 772994 (IG evals 772992, 772995). Proprio stays at one training seed.
  step8_auto.sh (detached): waits for those jobs, then 72 SDX zero-shots runs/zs_mu{1.0,0.5,1.5}_{fv4,cv4,trand,btactile}_t{1,2}_s{42,43,44}. Writes step8_auto.DONE or step8_auto.FAILED; no resubmission.
- Fine-tuning design 2026-11-02 (courseproject/FINETUNE_DESIGN_20261102.md; official code in penspin-16831/real). Open-loop replay of teacher records seeds 100/101 in SDX mu=1 with the pen fix (tools/collect_replays.py, runs/replays): 119 attempts, mean 31 steps vs IG 305, rot 0.34 rad vs 10.06; 0/119 reach pi or 2pi, so strict PenSpin filtering yields no demos at any budget. Teacher closed loop on eval seeds at mu=1: 54/192 >= pi, 41/192 >= 2pi; students at most 9/192 >= pi, 0 >= 2pi. Note: f and calib budget 100 also used 241 student replays, so it is not an equal-attempt budget.
- Fine-tuning (vi) done 2026-11-02 03:49 CST (user chose closed-loop oracle; results/superdex_finetune.json). Oracle attempts tools/collect_oracle.py, runs/oracle: 119 attempts, 10/25/50/100 give 2/5/11/22 demos >= 2pi. tools/finetune_student.py: official L1, Adam 1e-3, batch 512, 3000 epochs, all parameters, final epoch. Array 789917 (24 tasks), step9_auto.DONE, 144/144 JSON, .err empty.
  Proprio base mu=1 11.7±0.9. 2pi filter mu=0.5/1.0/1.5: b10 3.7/4.9/4.9, b25 4.1/5.0/5.2, b50 4.0/4.7/4.7, b100 3.9/4.6/4.2. Pi filter mu=1: 6.6-7.0. Post hoc b100 2pi mu=1 (job 790146, runs/ft_diag): epoch 400 5.2±0.8, epoch 400 lr 1e-4 7.6±0.7. All below base, 0 episodes >= 2pi. Val L1 0.18 → 0.09.
  → BC on closed-loop oracle states lowers closed-loop reward at every budget. Official open-loop fine-tuning itself is untested (0 demos).
- Training seeds done 2026-11-02 05:33 CST (results/superdex_step8_seeds.json). 789400-789412 COMPLETED, step8_auto.DONE, 72/72 JSON, .err empty. Mean ± sd over 3 training seeds (each the mean of 3 eval seeds):
  IG: B+tactile 42.9±2.3 (5.40 s), calib 35.4±3.7 (4.66 s), rand 26.1±0.9 (3.73 s), f 25.7±0.5 (3.65 s).
  SDX mu=0.5/1.0/1.5: B+tactile 12.6±2.2 / 15.7±2.0 / 15.7±1.6; rand 13.1±0.3 / 16.6±1.5 / 16.6±1.0; calib 13.8±0.6 / 17.1±0.9 / 18.3±0.9; f 13.1±1.1 / 15.9±1.6 / 16.1±1.5. Duration at mu=1: 2.46/2.63/2.57/2.54 s. Proprio (1 seed) 10.6/11.7/13.2. >=pi at mu=1 of 576: 22/23/18/24; >=2pi 0 for all.
  → The four tactile students cannot be separated at mu=1 (15.7-17.1, gaps within training-seed sd). The seed-0 claim "f below all simple baselines" does not hold with 3 seeds; f also does not beat them. All beat proprio.
- IG decomposition done 2026-11-06 01:30 CST (courseproject/DECOMPOSITION_DESIGN_20261105.md, rules fixed before submission; results/ig_decomposition.json). B+tactile 3 training seeds evaluated in IG with the student tactile channel raw / f v4 / calib v4. Jobs raw 802604/802610/802613, f 802606/802611/802614, calib 802608/802612/802615, all COMPLETED. Raw reproduces 758851/772992/772995 exactly.
  IG rot: raw 42.9±2.2 (5.51 s), calib 29.4±0.3 (4.06 s), f 18.3±0.1 (2.82 s); SDX mu=1 15.7. Observation share s: f 0.90, calib 0.50. f student under f in IG 25.7±0.5 vs B+tactile under f 18.3.
  → Observation gap dominates the B+tactile loss (f sensing alone takes it to within 2.6 of SDX), and training with f helps under f sensing (+7.4). The f student then loses 9.8 from IG to SDX (dynamics or f error on its own states, not separated). Possible floor effect near 16.
- Own-state f accuracy 2026-11-06 (design section 6, rule fixed beforehand; results/own_state_f1.json). Resync pairs (collect_pairs.py penlen=ig, mu=1) in runs/pairs_own for records 783809 (f student) and 783810 (calib student), eval seeds 42-44; B+tactile eval records 777024 lack qd (assert), so its control is held-out pairBT seed 103. Micro F1 ex pen, identity / calib / f: f student 0.639/0.659/0.721; calib student 0.649/0.679/0.749; B+tactile s103 0.646/0.685/0.775.
  → Gap Fo-Ff 0.0285, just under the 0.03 "dynamics more likely" bound; vs B+tactile alone 0.054. Marginal: f degrades somewhat on its own student's states (recall 0.68) and may explain part of the 9.8; split not resolved.
- f v5 2026-11-06 (courseproject/FV5_DESIGN_20261106.md, rules fixed before recording). IG record 803304: f v4 student 783805 with f v4 channel, seeds 109/110 (train_f) and 111 (shift_f), policy=student. fv5_auto.sh: 6/6 pairs (runs/pairs_v5), v4 rows identical in dataset_mu1_v5, train_f same settings + extra train_f. v4 config retrained on v5 data reproduces 0.826/0.775/0.737.
  F1 val/shift/shift_p/shift_f: f v4 0.826/0.775/0.737/0.739, f v5 s0 0.821/0.780/0.749/0.749. f student eval-seed pairs: 0.721 -> 0.755 (gap to other students 0.0285 -> 0.014). Port test 0 mismatches.
  Repo penspin-student-fv5-20261106 (clone of penlen 1923a67) commit d81ba4d, manifest 8aa1a5a7. Train 803318/803320/803322 (TRAIN_SEED 0/1/2), IG eval 803319/803321/803323. step10_auto.sh (detached): 27 SDX runs runs/zs_mu*_fv5_t{0,1,2}_s4?; DONE/FAILED marker, no resubmission.
- f v5 students done 2026-11-07 01:56 CST (results/superdex_step10_fv5.json). 803318-803323 COMPLETED, step10_auto.DONE, code sha check OK, 27/27 SDX JSON, .err empty.
  IG 24.7±1.4 (3.50 s) vs f v4 25.7±0.5. SDX mu=0.5/1.0/1.5: 13.4±1.0 / 15.3±0.8 / 15.4±0.6 (duration at mu=1 2.34 s; >=pi 16/576, >=2pi 0).
  Rule at mu=1: vs f v4 d=-0.60, 2SE=2.08 → not separable; vs calib v4 d=-1.74, 2SE=1.35 → f v5 below calib.
  → Better f on the f student's own states (F1 0.721 -> 0.755) does not help in SDX; IG-to-SDX loss 9.4 vs 9.7. f error on own states does not explain the loss.
- Proposal review and E1-E3 2026-11-07 (courseproject/PROPOSAL_REVIEW_20261107.md, rules fixed before running; results/e123_compliance_budget_20261107.json). Dir penspin-superdex-compliance-20261107: tools copy plus SDX_CONTACT override, pen contact distance audit per substep, joint-limit excess audit; CODE_SHA256 frozen; C0 reproduces the teacher JSONs 192/192.
  E1 teacher mu=1 (IG 136.3): C0 49.2±3.7, C1 (penalty 1e8) 32.7±3.7, C2 (1e10) 51.9±7.5, C3 (smoothing 1 mm) 52.7±5.6, C4 (falloff 1e-3) 18.8±1.5, C5 (C3+C4) 20.3±3.1. Max penetration 2.07-5.80 mm in every config (C0 3.05 mm); joint excess up to 8.5e-3 rad in C0. Gates (1 mm, 1e-6 rad) fail for all configs including C0, so stage 2 (students) was not run; no config is above C0 by 2 SE either.
  E2a budgets (val/shift/shift_p/shift_f micro F1): teacher-only f b10 0.737/0.581/0.518/0.546 → b100 0.823/0.711/0.654/0.653; calib b100 0.686/0.677/0.657/0.652; f + student replays b100 0.827/0.777/0.739/0.741.
  E2b onsets (±5 steps): f matches 0.37-0.46 of SDX onsets, 15-33 ms late, few extra onsets; calib 0.54-0.55, unbiased. E2c: open-loop pen error median 0.67 cm at 10-20 steps, 1.68 cm at 20-50, 6 cm after 50.
  E3 open-loop-label f: F1 0.11-0.50 at all budgets, below identity 0.63. All f results depend on privileged resync pairs.
- Gate decision 2026-11-07 (user: keep the 1 mm penetration and 1e-6 rad joint-limit gates for SuperDex). All SDX results so far (default contact) fail them. Cause: SDX joint limits are springs (100 N m/rad) and contact is soft penalty contact.
  Admissibility search (penspin-superdex-admissible-20261107, CODE_SHA256 frozen; SDX_JOINT and SDX_SUBSTEPS with zero-order-hold torque; unset reproduces 8/8). Rule fixed in PROPOSAL_REVIEW section 7. A(n) = limit stiffness 1e6, penalty 1e11, smoothing 0.2 mm. Teacher n=10/20/40: 47.2±5.1 / 28.7±0.8 / 12.7±0.7; max penetration 2.55/2.68/2.81 mm (2/3/4 of 192 episodes >1 mm); joint excess 1.6e-5/1.2e-4/4.0e-4 rad. Not converged (A10-A20 d=18.5 vs 2SE 5.9; A20-A40 d=16.0 vs 1.2).
  → No admissible, converged SDX setting within budget. Free-hand check: no torque accumulation across substeps, but joint trajectories still differ up to 0.17 rad (n=10 vs 80) near limits, so SDX at 5 ms carries large discretization error (inferred, not checked in SDX source). Per the proposal fallback, a different target simulator is needed for the comparison.
- MuJoCo target check 2026-11-07 (PROPOSAL_REVIEW section 8; dir penspin-mujoco-20261107, official mujoco 4.0.0 in its own venv, CODE_SHA256 frozen). Source stays Isaac Gym (PhysX); mjlab is not used. Hand = IG URDF (55 collision bodies, first <inertial> of link_3.0 as urdfdom, fixed links kept, no self-collision, gravcomp), IG PD/torque/friction/armature/limits, 32 tactile channels as in SDX. Kinematics check: reset fingertips match the IG record (<5e-6 m, 8 eps).
  Smoke (teacher, seed 42, 8 eps; IG 102.6, SDX default 21.3): MuJoCo default 87.0, penetration 3.38 mm, joint excess 1.8e-4. Stiffer contact/limits (tc 0.0005-0.002, d 0.99-0.9999, n 10-40): reward 32-66, joint excess 9e-6 to 1.2e-4 rad in 3-4/8 eps, penetration 1.27 mm overall and 0.23-1.12 mm after the first control step.
  IG initial states already overlap the pen in MuJoCo geometry (max 2.82 mm, 10/64 > 1 mm). IG's own records stay within 1e-7 rad of the limits (PhysX hard limits). MuJoCo soft limits overshoot on impact; 1e-6 rad would need ~1000x the compute. → MuJoCo also fails the 1e-6 rad joint gate; decision on gate definitions pending with user.
- Gates → reported metrics 2026-11-07 (user decision; finite checks stay hard; no clipping, no widened limits). MuJoCo added as a second target (courseproject/MUJOCO_DESIGN_20261107.md, rules fixed first).
  Target M = MuJoCo 4.0.0 default contact/limits, implicitfast, pyramidal, condim 3, mu=1. Teacher n=1/2/4/8: 97.7±18.9 / 100.2±11.6 / 101.5±14.7 / 100.8±16.3 (IG 136.3, SDX 49.2); n1-n2 d=-2.5 vs 2SE 25.6 → n=1. Teacher penetration max 4.57 mm (median 1.97), joint excess max 2.2e-3 rad.
  Zero-shot in M (3 training seeds): B+tactile 30.2±1.5, calib v4 30.3±1.2, rand 27.7±0.9, f v4 26.9±1.1, f v5 24.0±0.9, proprio 15.2 (1 seed). Students keep their IG ordering in M.
  M pairs (same IG records as v4, row-identical dataset): resync pen error median 0.047 cm (SDX 0.18); on-rate IG 8.0%, M 7.0%, SDX 4.8%. F1 val/shift/shift_p: identity 0.790/0.796/0.792, calib_M 0.800/0.798/0.790, f_M s0 0.848/0.838/0.830.
  Repo penspin-student-mjalign-20261107 (mj, 7140f67, manifest 328fab3e), port tests 0 mismatches. f_M train 807066/807068/807070 (IG eval 807067/807069/807071), calib_M 807072/807074/807076 (807073/807075/807077). mj_auto.sh (detached) runs the 18 M zero-shots after training; MJ_AUTO.DONE/FAILED, no resubmission.
