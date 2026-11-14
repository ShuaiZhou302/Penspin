# PenSpin: A Sim-to-Sim Study of Tactile Observation Alignment (CMU 16-831 Course Project)

**Mind the Touch Gap: Real-to-Sim Contact Alignment for Dexterous Pen Spinning**

This repository collects the project's code, design documents, experiment logs, and result summaries for review. The libraries for Isaac Gym (used for training) and SuperDex (the target simulator) are not included here (see Section 6).

## 1. One-sentence summary

Building on [PenSpin](https://github.com/HaozhiQi/penspin), we train a student policy with binary tactile input in Isaac Gym (IG, PhysX), then test it zero-shot in another simulator. There are two target simulators: SuperDex (primary target) and MuJoCo (second target).

Core idea: use a small number of paired replays to learn a model f that maps IG contact signals to the target simulator's contact signals, then re-distill the student in IG with the aligned tactile input. This changes only the student's observations, not the dynamics.

## 2. Main findings (as of 2026-11-11)

1. **Effective at the sensor level:** On SuperDex, f raises the micro F1 of IG→SDX contact from 0.64 to 0.83 (teacher states), and to 0.74–0.78 on student states; simple threshold/delay calibration only reaches 0.69.
2. **No improvement in transfer at the policy level:** On SuperDex at μ=1, all four tactile students (raw contact, randomization, calibration, f) fall within 15.7–17.1 and cannot be separated across 3 training seeds. They all score above the proprioception-only student (11.7), showing that touch itself helps, but the alignment quality does not show up.
3. **Alignment depends on privileged pairs:** f must be trained on per-step resync (state-matched, requires privileged state) pairs; when trained on deployable open-loop pairs, F1 is only 0.11–0.50, worse than no alignment.
4. **High F1 does not mean accurate timing:** f has high F1, but misses 55–60% of SDX contact onsets, and is on average 15–33 ms late.
5. **The SuperDex dynamics gap is large, and the default 5 ms step has not converged:** The teacher with privileged state drops from 136 in IG to 49; when each physics step is split finer, teacher reward keeps falling (10/20/40 substeps: 47→29→13).
6. **MuJoCo as the second target (default parameters, substep count converged):** The teacher retains 97.7. Students largely keep their IG ranking; the contact gap itself is much smaller (no-alignment F1 0.79, f_M 0.85). Students aligned specifically to MuJoCo are still training; **results pending**.

### Key numbers

μ=1.0; mean±sd is across 3 training seeds (each seed is first averaged over evaluation seeds 42–44 × 64 episodes).

| Student | IG | SuperDex | MuJoCo (zero-shot) |
|---|---|---|---|
| Teacher (privileged) | 136.3 | 49.2±3.7 | 97.7±18.9 (across eval seeds) |
| B+tactile (raw contact) | 42.9±2.3 | 15.7±2.0 | 30.2±1.5 |
| Calibration v4 | 35.4±3.7 | 17.1±0.9 | 30.3±1.2 |
| Contact randomization | 26.1±0.9 | 16.6±1.5 | 27.7±0.9 |
| f v4 (aligned to SDX) | 25.7±0.5 | 15.9±1.6 | 26.9±1.1 |
| f v5 (+ f-student pairs) | 24.7±1.4 | 15.3±0.8 | 24.0±0.9 |
| Proprio only (1 seed) | 14.2 | 11.7 | 15.2 |

Contact micro F1 (excluding the "pen total contact" channel), in the order val (teacher states) / shift (B+tactile student states) / shift_p (proprioception-only student states):

| Method | SuperDex | MuJoCo |
|---|---|---|
| No alignment | 0.636 / 0.646 / 0.631 | 0.790 / 0.796 / 0.792 |
| Threshold/delay calibration | 0.687 / 0.685 / 0.656 | 0.800 / 0.798 / 0.790 |
| f (budget 100 + student replays) | 0.826 / 0.775 / 0.737 | 0.848 / 0.838 / 0.830 |

More results and the decision rule for each step are in `docs/design/`.

## 3. Repository structure

```
docs/
  proposal/            2-page proposal (tex + pdf), references, pipeline figure
  design/              design + pre-fixed decision rules + results for each experiment (chronological)
    PROPOSAL_REVIEW_20261107.md   status, observations, and improvement plan for each proposal item, plus the compliance sweep and SuperDex/MuJoCo physics checks
    DECOMPOSITION_DESIGN_20261105.md  loss decomposition in IG (observation vs dynamics), accuracy of f on the student's own states
    FINETUNE_DESIGN_20261102.md       PenSpin-style target-domain fine-tuning baseline
    FV5_DESIGN_20261106.md            f v5 (one DAgger round adding f-student pairs)
    MUJOCO_DESIGN_20261107.md         MuJoCo as the second target simulator
  RUN_LOG.md           full experiment log (10-17 to 10-20 is mostly earlier WuJi reproduction; from 10-22 PenSpin students; from 10-28 SuperDex)
results/               per-stage result summary JSON (numbers in RUN_LOG and the design docs come from here)
models/                small models: teacher weights (numpy export), f v4/v5/f_M, calibration parameters, f fit results
code/
  student/             student training code: PenSpin upstream (MIT) + our patches, fully readable
    *_patch.py         per-feature patches (object pose estimation, tactile channels, f alignment, randomization/calibration baselines, noise, etc.)
    student_vs_penspin_upstream_5035c52.diff   changes relative to upstream PenSpin 5035c52
    GIT_HISTORY.txt, OTHER_STUDENT_REPOS_HISTORY.txt   commit history of each student repo
  isaac_recorder_patches/   git patches for the IG-side recorder (teacher/student replay recording, per-step velocity, tactile channels)
  superdex/
    tools/             SuperDex port and experiment tools (see Section 4)
    tools_diag/        decomposition, own-state accuracy, summary scripts
    scripts/           per-stage automation scripts (step7–10, rebuild, fv5, fine-tuning) and code SHA manifests
    jobs/              list of commands actually run in each stage
    compliance/        compliance sweep (only changed files + diff relative to the penlen version)
    admissible/        search for settings that "can pass the physics checks" (changed files + diff + diagnostic scripts)
  mujoco/
    tools/             MuJoCo port (mj_env.py, closed_loop_mj.py, collect_pairs_mj.py, etc.)
    scripts/, jobs/    run scripts and command lists
```

## 4. Pipeline and corresponding code

| Step | Content | Main code |
|---|---|---|
| 0 | Reproduce the PenSpin teacher (privileged, trained in IG) and the proprioception-only student | upstream PenSpin; `code/student/dagger_train.py` |
| 1 | Record teacher/student replays in IG (initial state, per-step q/qd/target/action/contact/pen pose) | `code/isaac_recorder_patches/` |
| 2 | SuperDex port: Allegro + pen, IG's PD control and 32-channel tactile | `code/superdex/tools/sdx_env.py`, `convert_urdf.py`, `teacher_np.py`, `student_policy.py`; port tests `test_*_port.py` |
| 3 | Paired data: per-step resync (state-matched) and open-loop replay | `collect_pairs.py`, `pair_stats.py` |
| 4 | Alignment model f and calibration baseline | `train_f.py`, `calib_thresh_delay.py`; open-loop version `compliance/train_f_open.py` |
| 5 | Re-distill students in IG with f/calibration/randomization tactile channels (DAgger, 400 iterations) | `code/student/falign_patch.py`, `tacbase_patch.py`, `scratch.sbatch` |
| 6 | Zero-shot evaluation: SuperDex / MuJoCo closed loop | `closed_loop.py`, `code/mujoco/tools/closed_loop_mj.py` |
| 7 | Diagnostics: loss decomposition, own-state accuracy, onset timing, divergence, compliance, step-size convergence | `tools_diag/`, `compliance/`, `admissible/` |
| 8 | Second target MuJoCo: pairs, f_M/calib_M, aligned students | `code/mujoco/` |

## 5. Where review help is most wanted

1. **Fidelity of the SuperDex port** (`sdx_env.py`): PD control (including the detail that in IG the D term is 0 on the first substep of each control step), joint limits, tactile computation (the resultant force of the pen on each link; in IG the index −1 channel is the pen's total contact force), pen length, and termination conditions.
2. **Whether f's training and evaluation have leakage or unfairness** (`train_f.py`): input features, data split (val/shift are both held-out recordings), and the definition of budget.
3. **Whether the student's f channel exactly matches the offline f** (`falign_patch.py` + `test_falign_port.py`; the logs show 0 mismatches).
4. **Whether the conclusions hold up:** especially "alignment improves F1 but not policy performance", and how the non-convergence of the SuperDex step size affects the conclusions.
5. **MuJoCo port** (`mj_env.py`): whether geometry, soft-contact parameters, and self-collision settings match IG.

## 6. Not included

- **Isaac Gym Preview 4:** an NVIDIA library; download it yourself.
- **SuperDex (Mochi physics engine):** not public; the code only contains the parts that call it.
- **Full PenSpin upstream repository:** only the files actually used by the student code are here; for the upstream license see `code/student/LICENSE_PenSpin_upstream_MIT.txt`.
- **Large files:** IG recordings (npz), student checkpoints, pair datasets, videos.
- **Machine-specific settings:** absolute paths, user and account names, and Slurm partitions are replaced with placeholders (`<PROJECT_ROOT>`, `<USER>`, `<ACCOUNT>`, `<PARTITION>`, `${MAIN_PARTITION}`, `${DEBUG_PARTITION}`). Set them for your own machine; the scripts are mainly for reading the logic.

## 7. Dependency versions

- **Isaac Gym side:** Isaac Gym Preview 4 (Python 3.8, PyTorch 2.2.2), PenSpin upstream 5035c52.
- **SuperDex side:** Python 3.12, NumPy 2.5.4, PyTorch 2.7.1 (CPU for student inference).
- **MuJoCo side:** `mujoco` 4.0.0 (official pip package).
