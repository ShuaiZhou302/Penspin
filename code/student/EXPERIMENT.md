# Student object-endpoint variants (pilot, 400 DAgger rounds)
Base: frozen strict pure-DAgger source penspin-16831-dagger-scratch-20261012 (STRICT_SOURCE_SHA256SUMS), same teacher705613 best_reward, seed0, beta0, 48x512 per round, reservoir196608, minibatch4096, LR1e-3, 16-action L1, random init.
Budget: 400 rounds = 9,830,400 labels (pilot; not the 59,006,976 full budget). Matched reference: 713476 iter_400_newsteps_9830400.pth, evaluated through this repo with input_mode=proprio.
A proprio_ends_gt: actor also gets simulator pen endpoints (official obj_ends: 3 frames, hand-root frame, official +/-pen_radius uniform noise) through the official end_feat_extractor MLP[6,6,6] with endpoint-symmetric max. NOT deployable proprio-only; upper-bound diagnostic.
B proprio_ends_est: an MLP[64,18] head on the student's own proprio transformer feature predicts the same 3-frame endpoints; prediction (not simulator ends) feeds the same extractor. Loss = BC L1 + POSE_LOSS_WEIGHT(1.0) * symmetric endpoint L1 (cm, mean per coordinate). Inputs at rollout/eval are proprio-only; asserted by obj_ends perturbation (zero action change).
Unchanged: reward, reset, physics, teacher, history transformer, 'proprio' code path. Different architecture/init RNG; not bitwise-paired trajectories. Evaluation: seeds42-44 x64, rotation and duration primary; custom full20+heading is not paper SR.

## Endpoint noise sensitivity (commit after 135803b)
ENDS_NOISE_CM in {1,2,3} for proprio_ends_gt only: i.i.d. Gaussian on all 18 endpoint-history inputs, resampled each control step and each minibatch (training) and each eval step, sigma = cm/100/sqrt(2/pi) so mean absolute per-coordinate error = ENDS_NOISE_CM, matching B's pose_l1_cm metric (B plateaued ~2.2 cm). Added on top of the official +/-pen_radius noise. Dedicated torch generators (train 24001, eval 24002) leave global CPU/CUDA RNG untouched. Default 0 reproduces A. Reservoir stores clean endpoints; noise applied at use. I.i.d. noise is not the same as an estimator's temporally correlated/biased error; this measures tolerance to unbiased error only.

## Estimator with tactile (proprio_tactile_ends_est)
Same as B (proprio_ends_est) but the endpoint estimator also reads the env binary tactile history (obs tactile_hist, 30x32, official contact threshold .05, 0.25 latency and dropout sensor model) through its own TemporalTransformer(32,2,2,32); concatenated with the proprio transformer feature into MLP[64,18]. Tactile reaches the policy only through the 18-d endpoint estimate; policy input dim 154 as B. Parameters 804,743 vs B 526,599 (larger estimator: capacity confound noted). Same loss/weight, 400 rounds. Reservoir stores tactile history.

## Controls for the tactile estimator gain
proprio_tactile_ends_est_sg: identical architecture/params (804,743) to proprio_tactile_ends_est, but the estimate is detached before the policy (end_feat_extractor input), so ends_head+tactile_tconv receive only the pose loss; tactile can inform the policy only as a pose estimate. Checked: BC gradient to estimator = 0, pose gradient > 0.
proprio_extra_ends_est: B plus an extra TemporalTransformer(32,2,2,32) on proprio_hist instead of tactile, same MLP[64,18] head -> 804,743 params, no tactile. Isolates estimator capacity.
Both 400 rounds, same loss/weight/seed/teacher as the other arms.

## Training-seed replication
TRAIN_SEED env (default 0 = all earlier runs) sets train.py seed and reservoir sampler seed 13001+TRAIN_SEED. A (proprio_ends_gt, no extra noise) and B+tactile (proprio_tactile_ends_est) get training seeds 1 and 2, 400 rounds each, same everything else; evaluation seeds stay 42-44 x64.

## MuJoCo-target alignment (second target simulator)
f_models/f_M_tps_b100_s0.pt = penspin-mujoco-20261107 runs/f_M/f_b100_s0.pt and tactile_calib/calib_M.npz = runs/calib_M/calib_M.npz:
same train_f.py / calib_thresh_delay.py settings as f v4 / calib v4 (budget 100 + train_bt, train_p), fit on resync pairs
collected in MuJoCo 4.0.0 (default contact, n=1) from the same Isaac Gym records as pairs_v4. Student code unchanged;
only the f/calib files and sbatch paths differ from penlen 1923a67. Port tests: f 0 mismatches / 203,776, calib 0.
