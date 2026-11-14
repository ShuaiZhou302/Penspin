# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://[arxiv link]
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import random
import numpy as np
import torch
import torch.nn as nn
from .pointnets import PointNet

from .block import TemporalConv, TemporalTransformer


# Verified priv_info layout for the PenSpin Allegro config (61 dims) and leave-one-group-out groups.
PRIV_LAYOUT = {'obj_position': (0, 3), 'obj_scale': (3, 4), 'obj_mass': (4, 5), 'obj_friction': (5, 6), 'obj_com': (6, 9),
               'obj_orientation': (9, 13), 'obj_angvel': (13, 16), 'fingertip_position': (16, 28),
               'obj_restitution': (28, 29), 'tactile': (29, 61)}
PRIV_DROP_GROUPS = {'objpose': ('obj_position', 'obj_orientation'), 'angvel': ('obj_angvel',),
                    'phys': ('obj_scale', 'obj_mass', 'obj_friction', 'obj_com', 'obj_restitution'),
                    'ftpos': ('fingertip_position',), 'contact': ('tactile',), 'pc': ()}


def priv_drop_group(input_mode):
    """None for proprio_priv_gt, the group name for proprio_priv_gt_no_<group>; asserts on anything else."""
    if input_mode == 'proprio_priv_gt':
        return None
    group = input_mode[len('proprio_priv_gt_no_'):]
    assert input_mode.startswith('proprio_priv_gt_no_') and group in PRIV_DROP_GROUPS, input_mode
    return group


class MLP(nn.Module):
    def __init__(self, units, input_size, with_last_activation=True):
        super(MLP, self).__init__()
        # use with_last_activation=False when we need the network to output raw values before activation
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        if not with_last_activation:
            layers.pop()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        policy_input_dim = kwargs.get('input_shape')[0]
        actions_num = kwargs.get('actions_num')
        self.units = kwargs.get('actor_units')
        self.priv_mlp = kwargs.get('priv_mlp_units')
        self.use_point_cloud_info = kwargs.get('use_point_cloud_info')
        self.use_fine_contact = kwargs.get('use_fine_contact')
        self.point_mlp_units = kwargs.get('point_mlp_units')
        self.use_point_transformer = kwargs.get('use_point_transformer')
        self.contact_mlp_units = kwargs.get('contact_mlp_units')
        self.contact_distillation = kwargs.get('contact_distillation', False)
        self.separate_temporal_fusion = kwargs.get('separate_temporal_fusion')
        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['proprio_adapt']
        self.proprio_mode = kwargs.get('proprio_mode', False)
        self.input_mode = kwargs.get('input_mode')
        self.proprio_len = kwargs.get('proprio_len', 30)
        self.student = kwargs.get('student', False)
        self.priv_mode = str(self.input_mode).startswith('proprio_priv_gt')
        self.proprio_family = self.priv_mode or self.input_mode in ('proprio', 'proprio_ends_gt', 'proprio_ends_est',
                                                  'proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg', 'proprio_extra_ends_est')

        if self.priv_info:
            policy_input_dim += self.priv_mlp[-1]
            # the output of env_mlp and proprioceptive regression should both be before activation
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'], with_last_activation=False)
            if self.priv_info_stage2:
                if self.separate_temporal_fusion:
                    # only proprioception is encoded
                    temporal_fusing_input_dim = 32
                    temporal_fusing_output_dim = 32
                else:
                    temporal_fusing_input_dim = 32
                    if self.contact_distillation:
                        temporal_fusing_input_dim += 32
                    temporal_fusing_output_dim = 8
                    if self.use_point_cloud_info:
                        temporal_fusing_output_dim += 32
                use_temporal_transformer = kwargs.get('use_temporal_transformer')
                use_position_encoder = kwargs.get('use_position_encoder')
                if use_temporal_transformer:
                    self.adapt_tconv = TemporalTransformer(temporal_fusing_input_dim, 2, 2, temporal_fusing_output_dim, use_pe=use_position_encoder)
                else:
                    self.adapt_tconv = TemporalConv(temporal_fusing_input_dim, temporal_fusing_output_dim)

        if self.proprio_mode and self.student:
            self.adapt_tconv = TemporalTransformer(32, 2, 2, 32, use_pe=True)
            # self.adapt_tconv = TemporalTransformer(16, 2, 2, 16, use_pe=True)
            # self.adapt_tconv = TemporalConv(16,16)
            self.all_fuse = nn.Linear(32, 40)
            proprio_feat_dim = 40
            policy_input_dim += proprio_feat_dim

        if "ends" in self.input_mode and self.student:
            policy_input_dim += 18
            self.end_feat_extractor = MLP(units=[6,6,6], input_size=3, with_last_activation=False)
        if self.input_mode == 'proprio_ends_est' and self.student:
            # Estimates the 3-frame endpoint history from the proprio transformer feature only.
            self.ends_head = MLP(units=[64, 18], input_size=32, with_last_activation=False)
        if self.input_mode in ('proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg') and self.student:
            # Estimator also reads binary tactile history; tactile reaches the policy only through the estimate.
            self.tactile_tconv = TemporalTransformer(32, 2, 2, 32, use_pe=True)
            self.ends_head = MLP(units=[64, 18], input_size=64, with_last_activation=False)
        if self.input_mode == 'proprio_extra_ends_est' and self.student:
            # Equal-capacity control: second proprio transformer in place of the tactile encoder.
            self.extra_tconv = TemporalTransformer(32, 2, 2, 32, use_pe=True)
            self.ends_head = MLP(units=[64, 18], input_size=64, with_last_activation=False)

        if self.priv_mode and self.student:
            self.priv_drop = priv_drop_group(self.input_mode)
            keep = torch.ones(kwargs['priv_info_dim'])
            for name in PRIV_DROP_GROUPS[self.priv_drop] if self.priv_drop else ():
                keep[slice(*PRIV_LAYOUT[name])] = 0.
            self.register_buffer('priv_keep', keep, persistent=False)
            # Upper-bound diagnostic: fresh encoders for the teacher's privileged inputs (not deployable).
            self.priv_enc = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'], with_last_activation=False)
            self.priv_point_mlp = MLP(units=self.point_mlp_units, input_size=3)
            policy_input_dim += self.priv_mlp[-1] + self.point_mlp_units[-1]

        if self.student and not self.proprio_family:
            # Other student modalities retain the original architecture.
            new_priv_dim = 12
            if self.use_point_cloud_info:
                policy_input_dim += 256
                self.pc_encoder = PointNet(point_channel=6)
            if "tactile" in self.input_mode:
                new_priv_dim += 32
            policy_input_dim += self.priv_mlp[-1]
            # policy_input_dim += 12
            self.env_mlp = MLP(units=self.priv_mlp, input_size=new_priv_dim, with_last_activation=False)

        if self.use_point_cloud_info and not self.student:
            policy_input_dim += self.point_mlp_units[-1]
            if self.use_point_transformer:
                self.point_mlp = TemporalTransformer(32, 2, 1, 32, use_pe=False, pre_ffn=True, input_dim=3)
            else:
                self.point_mlp = MLP(units=self.point_mlp_units, input_size=3)

        if self.contact_distillation:
            self.contact_mlp_s2 = MLP(units=self.contact_mlp_units, input_size=7)

        self.asymm_actor_critic = kwargs['asymm_actor_critic']
        self.critic_info_dim = kwargs['critic_info_dim']
        self.actor_mlp = MLP(units=self.units, input_size=policy_input_dim)
        self.value = MLP(units=self.units + [1], input_size=policy_input_dim + self.critic_info_dim) \
            if self.asymm_actor_critic else torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, extrin, extrin_gt = self._actor_critic(obs_dict)
        return mu, extrin, extrin_gt

    def _privileged_pred(self, joint_x, visual_x, tactile_x):
        # three part: modality specific transform, cross modality fusion, and temporal fusion
        # the order of the last two can be changed
        batch_dim, t_dim = joint_x.shape[:2]
        # ---- modality specific transform [*_x to *_t]
        joint_t = joint_x
        visual_t = None

        tactile_t = None
        if self.contact_distillation:
            contact_feat = self.contact_mlp_s2(tactile_x)
            valid_mask = tactile_x[..., [-1]] >= 0
            tactile_t = torch.sum(contact_feat * valid_mask, dim=2) / (torch.sum(valid_mask, dim=2) + 1e-9)

        # ---- cross modality fusion, and temporal fusion
        if self.separate_temporal_fusion:
            # temporal fusion first and then cross modality
            joint_t_t = self.adapt_tconv(joint_t)
            joint_visual_t_t = joint_t_t
            extrin_pred = self.all_fuse(joint_visual_t_t)
        else:
            # cross modality first and then temporal fusion
            # if the visual feature updates asynchronously, the temporal dimension would not match
            info_list = [joint_t]
            if self.contact_distillation:
                info_list.append(tactile_t)
            merge_t_t = torch.cat(info_list, dim=-1)
            extrin_pred = self.adapt_tconv(merge_t_t)

        return extrin_pred

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self._privileged_pred(obs_dict['proprio_hist'], obs_dict['depth_buf'], obs_dict['fine_contact_info'])
                # during supervised training, extrin has gt label
                if 'priv_info' in obs_dict:
                    extrin_gt = self.env_mlp(obs_dict['priv_info'])
                    if self.use_point_cloud_info:
                        pcs_gt = self.point_mlp(obs_dict['point_cloud_info'])
                        pcs_gt = torch.max(pcs_gt, 1)[0]
                        # 8 + 32 dimension
                        extrin_gt = torch.cat([extrin_gt, pcs_gt], dim=-1)
                else:
                    extrin_gt = extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs_input = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'])
                extrin_gt = extrin
            
        if self.use_point_cloud_info:
            if self.use_point_transformer:
                pcs = self.point_mlp(obs_dict['point_cloud_info'])
            else:
                pcs = self.point_mlp(obs_dict['point_cloud_info'])
                pcs = torch.max(pcs, 1)[0]
            if not self.student:
                extrin = torch.cat([extrin, pcs], dim=-1)
        
        ends_pred = None
        if self.proprio_mode and self.student:
            proprio_hidden = self.adapt_tconv(obs_dict['proprio_hist'])
            if self.input_mode == 'proprio_ends_est':
                ends_pred = self.ends_head(proprio_hidden).view(obs.shape[0], 3, 6)
            elif self.input_mode in ('proprio_tactile_ends_est', 'proprio_tactile_ends_est_sg'):
                est_hidden = torch.cat([proprio_hidden, self.tactile_tconv(obs_dict['tactile_hist'])], dim=-1)
                ends_pred = self.ends_head(est_hidden).view(obs.shape[0], 3, 6)
            elif self.input_mode == 'proprio_extra_ends_est':
                est_hidden = torch.cat([proprio_hidden, self.extra_tconv(obs_dict['proprio_hist'])], dim=-1)
                ends_pred = self.ends_head(est_hidden).view(obs.shape[0], 3, 6)
            proprio_feat = self.all_fuse(proprio_hidden)
            extrin = torch.cat([extrin,proprio_feat], dim=-1) if extrin is not None else proprio_feat
            if self.priv_mode:
                if self.priv_drop == 'pc':
                    pcs = obs.new_zeros(obs.shape[0], self.point_mlp_units[-1])
                else:
                    pcs = torch.max(self.priv_point_mlp(obs_dict['point_cloud_info']), 1)[0]
                extrin = torch.cat([extrin, self.priv_enc(obs_dict['priv_info'] * self.priv_keep), pcs], dim=-1)
            
        if "ends" in self.input_mode and self.student:
            if ends_pred is None:
                ends = obs_dict['obj_ends']
            else:
                ends = ends_pred.detach() if self.input_mode.endswith('_sg') else ends_pred
            point_feat_1 = self.end_feat_extractor(ends[..., :3]).unsqueeze(-1)
            point_feat_2 = self.end_feat_extractor(ends[..., 3:]).unsqueeze(-1)
            point_feat = torch.cat([point_feat_1, point_feat_2], dim=-1)
            point_feat = torch.max(point_feat, dim=-1)[0]
            extrin = torch.cat([extrin, point_feat.view(obs.shape[0], -1)], dim=-1)

        if not self.student:
            extrin = torch.tanh(extrin)
            obs_input = torch.cat([obs, extrin], dim=-1)
        elif self.proprio_family:
            # The proprioceptive student must not read simulator-only fingertip
            # positions from priv_info. Its actor input is obs plus history.
            extrin_gt = ends_pred.reshape(obs.shape[0], -1) if ends_pred is not None else extrin.detach()
            obs_input = torch.cat([obs, torch.tanh(extrin)], dim=-1)
        else:
            # extrin_gt = self.extrin_priv_mlp(obs_dict['priv_info'])

            if self.use_point_cloud_info:
                pc_embedding, self.point_indices = self.pc_encoder(obs_dict['student_pc_info'])
                extrin = pc_embedding
                obs_input = torch.cat([obs, pc_embedding], dim=-1)
            # fingertip_pose
            fingertip_pos = obs_dict['priv_info'][..., 16:28]
            new_priv = fingertip_pos + ((torch.rand(fingertip_pos.shape[0], 12).cuda()) - 0.5) * 0.02
           
            if "tactile" in self.input_mode:
                tactile = obs_dict['tactile_hist'][:,:-1,:].squeeze(1)
                new_priv = torch.cat([new_priv,tactile], dim=-1)
            new_priv = self.env_mlp(new_priv)
            extrin_gt = torch.tanh(new_priv)

            proprio_feat = torch.tanh(extrin)
            obs_input = torch.cat([obs, proprio_feat, extrin_gt], dim=-1)
        
        x = self.actor_mlp(obs_input)
        critic_obs = torch.cat([obs, obs_dict['critic_info']], dim=-1) if self.asymm_actor_critic else x
        value = self.value(critic_obs)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        if prev_actions is not None:
            prev_neglogp = -distr.log_prob(prev_actions).sum(1)
            prev_neglogp = torch.squeeze(prev_neglogp)
        else:
            prev_neglogp = None
        result = {
            'prev_neglogp': prev_neglogp,
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result
