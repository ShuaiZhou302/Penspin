"""Matched BC/DAgger continuation using existing PenSpin models and environment.

DAgger executes student means (beta=0) and queries the frozen oracle at every
visited state. Random-priority reservoir uniformly retains historical labeled
states across iterations. BC control differs only in the executing policy.
"""
import json
import hashlib
import math
import os
import time
from pathlib import Path
import torch


def ends_l1_cm(pred, target):
    """Mean per-coordinate L1 (cm) over 3 frames, symmetric in endpoint order."""
    pred, target = pred.reshape(-1, 3, 6), target.reshape(-1, 3, 6)
    a = (pred - target).abs().mean(-1)
    b = (pred - torch.cat([target[..., 3:], target[..., :3]], -1)).abs().mean(-1)
    return 100 * torch.minimum(a, b).mean()


class Reservoir:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.priority = torch.empty(0, dtype=torch.float64)
        self.data = {}
        self.seen = 0

    def add(self, batch):
        n = batch['obs'].shape[0]
        self.seen += n
        previous_size = len(self.priority)
        scores = torch.rand(n, generator=self.generator, dtype=torch.float64)
        scores = torch.cat((self.priority, scores))
        keep = scores.topk(min(self.capacity, len(scores)), sorted=False).indices
        self.retained_old = int((keep < previous_size).sum())
        self.data = {k: torch.cat((self.data[k], v.cpu()), 0)[keep] if k in self.data
                     else v.cpu()[keep] for k, v in batch.items()}
        self.priority = scores[keep]

    def sample(self, n, device):
        ids = torch.randint(len(self.priority), (n,), generator=self.generator)
        return {k: v[ids].to(device) for k, v in self.data.items()}


def run(agent):
    mode = os.environ['CONTINUE_MODE']
    assert mode in ('dagger', 'bc')
    input_mode = agent.input_mode
    from penspin.algo.models.models import PRIV_LAYOUT, priv_drop_group
    use_priv = input_mode.startswith('proprio_priv_gt')
    assert use_priv or input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est', 'proprio_tactile_ends_est',
                                      'proprio_tactile_ends_est_sg', 'proprio_extra_ends_est'), input_mode
    if use_priv:
        priv_drop_group(input_mode)
        layout = {k: tuple(v) for k, v in agent.env.priv_info_dict.items()}
        assert layout == PRIV_LAYOUT and agent.env.priv_info_dim == 61, layout
    use_ends = 'ends' in input_mode
    estimate = input_mode in ('proprio_ends_est', 'proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg', 'proprio_extra_ends_est')
    use_tactile = input_mode in ('proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg')
    pose_weight = float(os.environ.get('POSE_LOSS_WEIGHT', '1.0'))
    ends_noise_cm = float(os.environ.get('ENDS_NOISE_CM', '0'))
    assert ends_noise_cm >= 0 and (ends_noise_cm == 0 or input_mode == 'proprio_ends_gt'), ends_noise_cm
    noise_sigma = ends_noise_cm / 100 / math.sqrt(2 / math.pi)
    noise_gen = torch.Generator(device=agent.device)
    noise_gen.manual_seed(24001)
    def add_ends_noise(x):
        if noise_sigma == 0:
            return x
        return x + noise_sigma * torch.randn(x.shape, generator=noise_gen, device=x.device)
    if ends_noise_cm > 0:
        probe = add_ends_noise(torch.zeros(4096, 3, 6, device=agent.device))
        print('ENDS_NOISE', json.dumps({'target_mean_abs_cm': ends_noise_cm, 'sigma_m': noise_sigma,
              'measured_mean_abs_cm': float(100 * probe.abs().mean())}), flush=True)
    print('OBJPOSE_CONFIG', json.dumps({'input_mode': input_mode, 'pose_loss_weight': pose_weight if estimate else None}), flush=True)
    iterations = int(os.environ.get('CONTINUE_ITERS', '100'))
    capacity = int(os.environ.get('RESERVOIR_CAPACITY', '196608'))
    env = agent.env
    n, horizon = env.num_envs, agent.horizon_length
    reservoir = Reservoir(capacity, 13001 + int(os.environ.get('TRAIN_SEED', '0')))
    for module in (agent.demon_model, agent.running_mean_std_demon,
                   agent.priv_mean_std_demon, agent.point_cloud_mean_std_demon):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    teacher_before = {k: v.detach().cpu().clone() for k, v in agent.demon_model.state_dict().items()}
    normalizer_before = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                         for m in (agent.running_mean_std_demon, agent.priv_mean_std_demon, agent.point_cloud_mean_std_demon)]
    start_iteration, steps = 0, 0
    resume_path = os.environ.get('CONTINUE_RESUME')
    if resume_path:
        state = torch.load(resume_path, map_location='cpu')
        assert state['mode'] == mode
        assert len(state['priority']) == min(capacity, state['seen'])
        assert state['seen'] == state['new_steps']
        agent.model.load_state_dict(state['model'], strict=True)
        agent.running_mean_std.load_state_dict(state['running_mean_std'], strict=True)
        agent.optimizer.load_state_dict(state['optimizer'])
        reservoir.data = state['reservoir']
        reservoir.priority = state['priority']
        reservoir.seen = state['seen']
        reservoir.generator.set_state(state['sampler_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        start_iteration, steps = state['iterations'], state['new_steps']
        assert start_iteration < iterations
        assert all(len(v) == len(reservoir.priority) for v in reservoir.data.values())
        assert all(torch.equal(v.detach().cpu(), state['model'][k]) for k,v in agent.model.state_dict().items())
        assert len(agent.optimizer.state) > 0
        print('RESUME_VERIFIED', json.dumps({'path':resume_path, 'mode':mode,
              'iteration':start_iteration, 'new_steps':steps, 'reservoir_seen':reservoir.seen,
              'reservoir_size':len(reservoir.priority), 'optimizer_states':len(agent.optimizer.state),
              'simulator_state_restored':False}), flush=True)
        del state
    if not resume_path:
        assert not agent.optimizer.state, 'scratch must have fresh optimizer'
        digest = hashlib.sha256()
        for key, value in sorted(agent.model.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        print('SCRATCH_INITIALIZATION', json.dumps({
            'model_sha256': digest.hexdigest(), 'optimizer_states': len(agent.optimizer.state),
            'student_checkpoint_loaded': False, 'student_normalizer_loaded': False,
            'normalizer_count': float(agent.running_mean_std.count),
            'reservoir_seen': reservoir.seen, 'teacher_is_pretrained': True}), flush=True)
        agent.save(str(Path(agent.nn_dir)/'initial_random'))
    agent.set_eval()
    obs = env.reset()
    bsize = n*horizon
    buffer = {
        'obs': torch.empty((bsize, obs['obs'].shape[-1]), device=agent.device),
        'proprio': torch.empty((bsize, agent.proprio_len, agent.proprio_dim), device=agent.device),
        'target': torch.empty((bsize, agent.actions_num), device=agent.device),
    }
    if use_ends:
        assert tuple(obs['obj_ends'].shape) == (n, 3, 6), obs['obj_ends'].shape
        buffer['ends'] = torch.empty((bsize, 3, 6), device=agent.device)
    if use_tactile:
        assert tuple(obs['tactile_hist'].shape) == (n, agent.proprio_len, 32), obs['tactile_hist'].shape
        buffer['tactile'] = torch.empty((bsize, agent.proprio_len, 32), device=agent.device)
    if use_priv:
        buffer['priv'] = torch.empty((bsize, obs['priv_info'].shape[-1]), device=agent.device)
        buffer['pc'] = torch.empty((bsize,) + tuple(obs['point_cloud_info'].shape[1:]), device=agent.device)
    out = Path(agent.nn_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out.parent/'continuation.jsonl'
    lengths = torch.zeros(n, device=agent.device)
    def save_resume_state(completed_iteration, completed_steps):
        state_path = out/'continuation_state.pt'
        torch.save({'model':agent.model.state_dict(),'running_mean_std':agent.running_mean_std.state_dict(),
                'optimizer':agent.optimizer.state_dict(),'reservoir':reservoir.data,
                'priority':reservoir.priority,'seen':reservoir.seen,
                'sampler_rng':reservoir.generator.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'new_steps':completed_steps,'iterations':completed_iteration,
                'mode':mode}, out/'continuation_state.tmp')
        os.replace(out/'continuation_state.tmp', state_path)
    started = time.time()
    for iteration in range(start_iteration+1, iterations+1):
        agent.set_eval()
        completed_lengths = []
        rollout_error, action_delta = 0., 0.
        rollout_pose = 0.
        for t in range(horizon):
            with torch.no_grad():
                history = obs['proprio_hist'][..., -agent.proprio_len:, :agent.proprio_dim]
                student_in = {'obs': agent.running_mean_std(obs['obs']), 'proprio_hist': history}
                pc = obs['point_cloud_info']
                if agent.normalize_point_cloud:
                    pc = agent.point_cloud_mean_std_demon(pc.reshape(-1,3)).reshape(n,-1,3)
                priv = agent.priv_mean_std_demon(obs['priv_info']) if agent.normalize_priv else obs['priv_info']
                if use_priv:
                    student_in['priv_info'], student_in['point_cloud_info'] = priv, pc
                if use_ends:
                    student_in['obj_ends'] = add_ends_noise(obs['obj_ends'])
                if use_tactile:
                    student_in['tactile_hist'] = obs['tactile_hist']
                student, student_extrin, student_aux = agent.model.act_inference(student_in)
                if iteration == start_iteration+1 and t == 0 and use_priv:
                    keep = agent.model.priv_keep
                    moved = dict(student_in); moved['priv_info'] = priv + 0.5 * keep
                    pdelta = float((agent.model.act_inference(moved)[0] - student).abs().max())
                    moved = dict(student_in); moved['priv_info'] = priv + 0.5 * (1 - keep)
                    ddelta = float((agent.model.act_inference(moved)[0] - student).abs().max())
                    moved = dict(student_in); moved['point_cloud_info'] = pc + 0.5
                    cdelta = float((agent.model.act_inference(moved)[0] - student).abs().max())
                    drop = agent.model.priv_drop
                    assert pdelta > 0. and ddelta == 0. and (cdelta == 0.) == (drop == 'pc'), ('priv dependency', pdelta, ddelta, cdelta)
                    print('PRIV_DEPENDENCY', json.dumps({'priv_max_action_delta': pdelta, 'dropped_priv_max_action_delta': ddelta,
                          'point_cloud_max_action_delta': cdelta, 'drop_group': drop, 'dropped_dims': int((keep == 0).sum()),
                          'priv_dim': priv.shape[-1], 'points': pc.shape[1]}), flush=True)
                if iteration == start_iteration+1 and t == 0 and use_ends:
                    moved = dict(student_in); moved['obj_ends'] = obs['obj_ends'] + 0.05
                    delta = float((agent.model.act_inference(moved)[0] - student).abs().max())
                    if estimate:
                        assert delta == 0., ('estimator variant read simulator obj_ends', delta)
                    else:
                        assert delta > 0., 'obj_ends input unused'
                    print('OBJ_ENDS_DEPENDENCY', json.dumps({'input_mode': input_mode, 'max_action_delta': delta}), flush=True)
                    if use_tactile:
                        flipped = dict(student_in); flipped['tactile_hist'] = 1 - obs['tactile_hist']
                        tdelta = float((agent.model.act_inference(flipped)[0] - student).abs().max())
                        assert tdelta > 0., 'tactile input unused'
                        print('TACTILE_DEPENDENCY', json.dumps({'max_action_delta': tdelta,
                              'tactile_on_fraction': float(obs['tactile_hist'].mean())}), flush=True)
                if estimate:
                    rollout_pose += float(ends_l1_cm(student_aux, obs['obj_ends']))
                teacher_in = {'obs': agent.running_mean_std_demon(obs['obs']), 'priv_info': priv,
                              'point_cloud_info': pc, 'proprio_hist': history}
                teacher, teacher_extrin, _ = agent.demon_model.act_inference(teacher_in)
                student, teacher = student.clamp(-1,1), teacher.clamp(-1,1)
                sl = slice(t*n, (t+1)*n)
                buffer['obs'][sl].copy_(obs['obs'])
                buffer['proprio'][sl].copy_(history)
                buffer['target'][sl].copy_(teacher)
                if use_ends:
                    buffer['ends'][sl].copy_(obs['obj_ends'])
                if use_tactile:
                    buffer['tactile'][sl].copy_(obs['tactile_hist'])
                if use_priv:
                    buffer['priv'][sl].copy_(priv)
                    buffer['pc'][sl].copy_(pc)
                actions = student if mode == 'dagger' else teacher
                extrin = student_extrin if mode == 'dagger' else teacher_extrin
                assert torch.isfinite(actions).all()
                rollout_error += float((student-teacher).abs().sum(-1).mean())
                action_delta = max(action_delta, float((actions-(student if mode=='dagger' else teacher)).abs().max()))
                obs, reward, done, _ = env.step(actions, extrin_record=extrin)
                lengths += 1
                ids = done.bool()
                completed_lengths.extend(lengths[ids].cpu().tolist())
                lengths[ids] = 0
        steps += bsize
        reservoir.add(buffer)
        agent.set_train()
        losses, pose_losses = [], []
        updates = agent.mini_epochs_num * max(1, bsize // agent.minibatch_size)
        for _ in range(updates):
            batch = reservoir.sample(agent.minibatch_size, agent.device)
            batch_in = {'obs': agent.running_mean_std(batch['obs']), 'proprio_hist': batch['proprio']}
            if use_ends:
                batch_in['obj_ends'] = add_ends_noise(batch['ends'])
            if use_tactile:
                batch_in['tactile_hist'] = batch['tactile']
            if use_priv:
                batch_in['priv_info'], batch_in['point_cloud_info'] = batch['priv'], batch['pc']
            mu, _, _, _, aux = agent.model._actor_critic(batch_in)
            bc_loss = (mu.clamp(-1,1)-batch['target']).abs().sum(-1).mean()
            loss = bc_loss
            if estimate:
                pose_loss = ends_l1_cm(aux, batch['ends'])
                loss = bc_loss + pose_weight * pose_loss
                if input_mode.endswith('_sg') and iteration == start_iteration+1 and not pose_losses:
                    head = list(agent.model.ends_head.parameters()) + list(agent.model.tactile_tconv.parameters())
                    g = torch.autograd.grad(bc_loss, head, retain_graph=True, allow_unused=True)
                    gn = sum(float(x.abs().sum()) for x in g if x is not None)
                    assert gn == 0., ('stop-gradient leak', gn)
                    gp = torch.autograd.grad(pose_loss, head, retain_graph=True, allow_unused=True)
                    gpn = sum(float(x.abs().sum()) for x in gp if x is not None)
                    assert gpn > 0.
                    print('SG_GRADIENT_CHECK', json.dumps({'bc_to_estimator': gn, 'pose_to_estimator': gpn}), flush=True)
                pose_losses.append(float(pose_loss.detach()))
            assert torch.isfinite(loss)
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.grad_norm)
            agent.optimizer.step()
            losses.append(float(bc_loss.detach()))
        row = {'iteration':iteration, 'mode':mode, 'new_labeled_steps':steps,
               'reservoir_size':len(reservoir.priority), 'reservoir_seen':reservoir.seen,
               'retained_previous_round_samples':reservoir.retained_old,
               'loss_l1_sum':sum(losses)/len(losses), 'input_mode':input_mode,
               'pose_l1_cm':sum(pose_losses)/len(pose_losses) if pose_losses else None,
               'rollout_pose_l1_cm':rollout_pose/horizon if estimate else None, 'rollout_l1_sum':rollout_error/horizon,
               'controller_max_difference':action_delta,
               'completed_training_episode_length_steps':sum(completed_lengths)/len(completed_lengths) if completed_lengths else None,
               'elapsed_s':time.time()-started,
               'gpu_peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20}
        with metrics_path.open('a') as file:
            file.write(json.dumps(row)+'\n')
        print('CONTINUATION', json.dumps(row), flush=True)
        if iteration in (100,400,500,1000,1500,2000,2001,2401) or iteration==iterations:
            agent.save(str(out/f'iter_{iteration}_newsteps_{steps}'))
        if iteration % 250 == 0:
            save_resume_state(iteration, steps)
    for k,v in agent.demon_model.state_dict().items():
        assert torch.equal(v.cpu(),teacher_before[k]), ('teacher changed',k)
    for m, before in zip((agent.running_mean_std_demon,agent.priv_mean_std_demon,agent.point_cloud_mean_std_demon),normalizer_before):
        assert all(torch.equal(v.cpu(),before[k]) for k,v in m.state_dict().items())
    agent.save(str(out/'final'))
    save_resume_state(iterations, steps)
    print('FROZEN_TEACHER_AND_NORMALIZERS_VERIFIED', flush=True)
