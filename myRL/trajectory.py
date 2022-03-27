import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class Trajectory(object):
    def __init__(self, num_steps, prop_shape,visual_shape,action_space):
        """
        params:
        num_steps:number of forward /horizen in one episode
        num_processes:cpu processes used to train
        prop_shape:proprioceptive observation space in envs
        visual_shape:visual infos obs space in envs
        action_shape:action space in envs
                 """
        #trajetory:(s1,a1,r1,s2,a2,r2,s3,a3,r3....sn,an,rn,s(n+1))
        action_shape=action_space.shape[0]
        self.obs_prop = torch.zeros(num_steps+1, *prop_shape)
        self.obs_visual=torch.zeros(num_steps+1,*visual_shape)
        self.actions = torch.zeros(num_steps, action_shape)
        #reward is the immediately reward for correspoding step
        self.rewards = torch.zeros(num_steps, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1)
        #returns is the long term value
        self.returns = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps, 1)
        
        self.masks = torch.ones(num_steps+1, 1)
        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps+1, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs_prop = self.obs_prop.to(device)
        self.obs_visual=self.obs_visual.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs_prop,obs_visual,actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs_prop[self.step + 1].copy_(obs_prop)
        self.obs_visual[self.step+1].copy_(obs_visual)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs_prop[0].copy_(self.obs_prop[-1])
        self.obs_visual[0].copy_(self.obs_visual[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        gamma,
                        use_proper_time_limits=False):
        #compute returns in reversed squence
        if use_proper_time_limits:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1]*gamma*self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format( num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            prop_batch = self.obs_prop[:-1].view(-1, *self.obs_prop.size()[1:])[indices]
            vision_batch = self.obs_visual[:-1].view(-1, *self.obs_visual.size()[1:])[indices]
            actions_batch = self.actions.view(-1,self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield prop_batch,vision_batch,actions_batch, \
                value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ

