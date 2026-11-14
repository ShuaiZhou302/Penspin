"""First-episode PenSpin audit. Instrumentation does not draw random numbers."""
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch


_ENDS_NOISE = {}


def _ends_noise(x):
    """ENDS_NOISE_CM evaluation noise (same definition as dagger_train); dedicated generator, global RNG untouched."""
    import math
    cm = float(os.environ.get("ENDS_NOISE_CM", "0"))
    if cm == 0:
        return x
    if "g" not in _ENDS_NOISE:
        _ENDS_NOISE["g"] = torch.Generator(device=x.device)
        _ENDS_NOISE["g"].manual_seed(24002)
        print("EVAL_ENDS_NOISE_CM", cm, flush=True)
    return x + cm / 100 / math.sqrt(2 / math.pi) * torch.randn(x.shape, generator=_ENDS_NOISE["g"], device=x.device)


def run(agent):
    env = agent.env
    n, device = env.num_envs, agent.device
    mode = os.environ.get("AUDIT_MODE", "baseline")
    label = os.environ["AUDIT_LABEL"]
    agent.set_eval()
    scales = env.priv_info_buf[:, env.priv_info_dict["obj_scale"][0]].clone()
    radii, lengths, names = [], [], []
    for i in range(n):
        key = env.object_type_list[int(env.object_type_at_env[i])]
        filename = env.asset_files_dict[key]
        cylinder = ET.parse(filename).find("./link/collision/geometry/cylinder")
        radii.append(float(cylinder.get("radius")))
        lengths.append(float(cylinder.get("length")))
        names.append(filename)
    lengths = torch.tensor(lengths, device=device) * scales
    radii = torch.tensor(radii, device=device) * scales
    legacy_length = float(env.pen_length * env.obj_scale)
    capture = {}
    original_reward = env.compute_reward
    original_termination = env.check_termination
    original_observations = env.compute_observations

    def causes(position, quaternion):
        # PenSpin cylinder's shaft is its local +Z axis (xyzw quaternion).
        qx, qy, qz, qw = quaternion.unbind(-1)
        shaft = torch.stack((2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx), 1-2*(qx*qx+qy*qy)), -1)
        extent = lengths * 0.5 * shaft[:, 2].abs()
        legacy_extent = legacy_length * 0.5 * shaft[:, 2].abs()
        center = position[:, 2] < env.reset_z_threshold
        endpoints = (position[:, 2]-extent < .56) | (position[:, 2]+extent > .71)
        old_endpoints = (position[:, 2]-legacy_extent < .56) | (position[:, 2]+legacy_extent > .71)
        timeout = env.progress_buf >= env.max_episode_length
        return center, endpoints, old_endpoints, timeout, shaft

    if mode == "per_env_geometry":
        def corrected_termination(position):
            c, e, old, t, shaft = causes(position, env.object_rot)
            return c | e | t
        env.check_termination = corrected_termination
    if mode == "reset_history":
        def corrected_observations():
            ids = env.at_reset_buf.nonzero(as_tuple=False).squeeze(-1).clone()
            original_observations()
            if len(ids):
                width = env.obs_buf.shape[1] // 3
                fresh = env.obs_buf_lag_history[ids, -3:, :width].reshape(len(ids), -1)
                env.obs_buf[ids, :fresh.shape[1]] = fresh
        env.compute_observations = corrected_observations

    def captured_reward(actions):
        original_reward(actions)
        c, e, old, t, shaft = causes(env.object_pos, env.object_rot)
        capture.update(center=c.clone(), endpoint=e.clone(), legacy_endpoint=old.clone(),
                       timeout=t.clone(), shaft=shaft.clone(), pose=torch.cat((env.object_pos, env.object_rot), -1).clone(),
                       progress=env.progress_buf.clone(), done=env.reset_buf.clone())
    env.compute_reward = captured_reward
    obs = env.reset()
    initial_progress = env.progress_buf.clone()
    initial_timeout = capture["timeout"].clone()
    initial_done = capture["done"].clone()
    # Existing evaluations start here, omitting reset()'s physical warmup step.
    finished = torch.zeros(n, device=device, dtype=torch.bool)
    totals = torch.zeros(n, device=device)
    rotations = totals.clone()
    rot_rewards = totals.clone()
    heading = totals.clone()
    counts = torch.zeros(n, device=device, dtype=torch.long)
    end_progress = counts.clone()
    term = {k: torch.zeros_like(finished) for k in ("center", "endpoint", "legacy_endpoint", "timeout")}
    ever180 = finished.clone()
    heading_invalid = counts.clone()
    q = env.object_rot
    _, _, _, _, shaft = causes(env.object_pos, q)
    old_heading = torch.atan2(shaft[:, 1], shaft[:, 0])
    old_horizontal = shaft[:, :2].norm(dim=-1)
    pose_frames, active_frames = [], []
    for step in range(env.max_episode_length + 1):
        with torch.no_grad():
            inputs = {"obs": agent.running_mean_std(obs["obs"]),
                      "proprio_hist": obs["proprio_hist"][..., -agent.proprio_len:, :agent.proprio_dim]}
            if agent.input_mode.startswith("proprio_priv_gt"):
                pc = obs["point_cloud_info"]
                if agent.normalize_point_cloud:
                    pc = agent.point_cloud_mean_std_demon(pc.reshape(-1, 3)).reshape(n, -1, 3)
                inputs["priv_info"] = agent.priv_mean_std_demon(obs["priv_info"]) if agent.normalize_priv else obs["priv_info"]
                inputs["point_cloud_info"] = pc
                if step == 0:
                    keep = agent.model.priv_keep
                    moved = dict(inputs); moved["priv_info"] = inputs["priv_info"] + 0.5 * keep
                    d = float((agent.model.act_inference(moved)[0] - agent.model.act_inference(inputs)[0]).abs().max())
                    moved = dict(inputs); moved["priv_info"] = inputs["priv_info"] + 0.5 * (1 - keep)
                    dd = float((agent.model.act_inference(moved)[0] - agent.model.act_inference(inputs)[0]).abs().max())
                    moved = dict(inputs); moved["point_cloud_info"] = inputs["point_cloud_info"] + 0.5
                    dc = float((agent.model.act_inference(moved)[0] - agent.model.act_inference(inputs)[0]).abs().max())
                    assert d > 0. and dd == 0. and (dc == 0.) == (agent.model.priv_drop == "pc"), (d, dd, dc)
                    print("EVAL_PRIV_DEPENDENCY", agent.input_mode, d, dd, dc, flush=True)
            if "ends" in agent.input_mode:
                inputs["obj_ends"] = _ends_noise(obs["obj_ends"])
                if "tactile" in agent.input_mode:
                    inputs["tactile_hist"] = obs["tactile_hist"]
                if step == 0:
                    moved = dict(inputs); moved["obj_ends"] = obs["obj_ends"] + 0.05
                    d = float((agent.model.act_inference(moved)[0] - agent.model.act_inference(inputs)[0]).abs().max())
                    assert (d == 0.) == ("ends_est" in agent.input_mode), (agent.input_mode, d)
                    print("EVAL_OBJ_ENDS_DEPENDENCY", agent.input_mode, d, flush=True)
            action, extrin, _ = agent.model.act_inference(inputs)
            obs, reward, done, info = env.step(action.clamp(-1, 1), extrin_record=extrin)
        active = ~finished
        totals += reward * active
        rotations += env.last_step_axis_rotation_rad * active
        rot_rewards += env.last_step_clipped_rotation_reward * active
        counts += active.long()
        shaft = capture["shaft"]
        current_heading = torch.atan2(shaft[:, 1], shaft[:, 0])
        delta = torch.atan2(torch.sin(current_heading-old_heading), torch.cos(current_heading-old_heading))
        valid = (shaft[:, :2].norm(dim=-1) > .1) & (old_horizontal > .1)
        heading += torch.where(active & valid, delta, torch.zeros_like(delta))
        heading_invalid += (active & ~valid).long()
        old_heading = current_heading
        old_horizontal = shaft[:, :2].norm(dim=-1)
        ever180 |= rotations >= np.pi
        ended = active & done.bool()
        for key in term:
            term[key][ended] = capture[key][ended]
        end_progress[ended] = capture["progress"][ended]
        pose_frames.append(capture["pose"].cpu().numpy())
        active_frames.append(active.cpu().numpy())
        finished |= ended
        if bool(finished.all()):
            break
    actual_fail = term["center"] | term["endpoint"]
    clean_timeout = term["timeout"] & ~actual_fail
    legacy_full = counts >= env.max_episode_length
    duration = (counts + initial_progress).float() * env.dt * env.control_freq_inv
    result = {
        "label": label, "mode": mode, "n": n, "finished": int(finished.sum()),
        "initial_progress_values": initial_progress.cpu().tolist(),
        "warmup_terminated_count": int(initial_done.sum()),
        "legacy_count_ge400": int(legacy_full.sum()),
        "actual_timeout_count": int(term["timeout"].sum()),
        "timeout_without_actual_geometry_failure": int(clean_timeout.sum()),
        "center_termination_count": int(term["center"].sum()),
        "actual_endpoint_termination_count": int(term["endpoint"].sum()),
        "legacy_endpoint_termination_count": int(term["legacy_endpoint"].sum()),
        "false_legacy_endpoint_count": int((term["legacy_endpoint"] & ~term["endpoint"] & ~term["center"] & ~term["timeout"]).sum()),
        "mean_total_reward_after_warmup": float(totals.mean()),
        "mean_rotation_reward_after_warmup": float(rot_rewards.mean()),
        "mean_old_duration_s": float(counts.float().mean() * env.dt * env.control_freq_inv),
        "mean_physical_duration_s": float(duration.mean()),
        "ever180_count": int(ever180.sum()),
        "timeout_and_integrated180": int((clean_timeout & (rotations >= np.pi)).sum()),
        "timeout_and_shaft_heading180": int((clean_timeout & (heading >= np.pi) & (heading_invalid == 0)).sum()),
        "mean_shaft_heading_rad": float(heading.mean()),
        "legacy_scalar_length_m": legacy_length,
        "actual_length_minmax_m": [float(lengths.min()), float(lengths.max())],
        "object_assets": names, "length_m": lengths.cpu().tolist(), "radius_m": radii.cpu().tolist(),
        "episode_steps_after_warmup": counts.cpu().tolist(), "end_progress": end_progress.cpu().tolist(),
        "timeout": term["timeout"].cpu().tolist(), "heading_rad": heading.cpu().tolist(),
        "rotation_rad": rotations.cpu().tolist(), "force_scale": env.force_scale,
    }
    out = Path("runs")
    out.mkdir(exist_ok=True)
    (out / (label + ".json")).write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / (label + "_poses.npz"), pose=np.stack(pose_frames), active=np.stack(active_frames))
    print("PROTOCOL_AUDIT", json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}), flush=True)

