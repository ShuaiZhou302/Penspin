"""PenSpin proprio-family student checkpoints run with torch on CPU inside the SuperDex environment.

The network code is imported unchanged from the frozen Isaac Gym copy (penspin.algo.models.models at fa99bdc+); inputs
mirror protocol_student_diagnostic: obs = student running_mean_std(obs96), proprio_hist (30, 32), tactile_hist (30, 32)
for tactile modes, obj_ends for ends modes (unused by estimator modes). Action = clamp(mu, -1, 1)."""
import sys
from pathlib import Path
import numpy as np
import torch

ISAAC = Path(__file__).resolve().parent.parent / 'isaac'
sys.path.insert(0, str(ISAAC))
from penspin.algo.models.models import ActorCritic  # noqa: E402
from penspin.algo.models.running_mean_std import RunningMeanStd  # noqa: E402

NET = dict(actor_units=[512, 256, 128], priv_mlp_units=[256, 128, 8], actions_num=16, input_shape=(96,), priv_info=False,
           proprio_adapt=False, priv_info_dim=61, critic_info_dim=0, asymm_actor_critic=False, point_mlp_units=[32, 32, 32],
           use_fine_contact=False, contact_mlp_units=[32, 32, 32], use_point_transformer=False, proprio_mode=True,
           proprio_len=30, student=True, use_point_cloud_info=False)


class StudentPolicy:
    def __init__(self, checkpoint, input_mode):
        torch.set_num_threads(1)
        ck = torch.load(checkpoint, map_location='cpu')
        self.mode = input_mode
        self.model = ActorCritic(dict(NET, input_mode=input_mode)).eval()
        self.model.load_state_dict(ck['model'], strict=True)
        self.rms = RunningMeanStd((96,)).eval()
        self.rms.load_state_dict(ck['running_mean_std'], strict=True)

    @torch.no_grad()
    def act(self, obs96, proprio_hist, tactile_hist):
        inp = {'obs': self.rms(torch.as_tensor(np.asarray(obs96, np.float32))[None]),
               'proprio_hist': torch.as_tensor(np.asarray(proprio_hist, np.float32))[None]}
        if self.mode != 'proprio':
            inp['obj_ends'] = torch.zeros(1, 3, 6)
        if 'tactile' in self.mode:
            inp['tactile_hist'] = torch.as_tensor(np.asarray(tactile_hist, np.float32))[None]
        return self.model.act_inference(inp)[0].clamp(-1, 1)[0].numpy()
