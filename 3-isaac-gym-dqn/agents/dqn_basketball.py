from envs.env_basketball import Basketball
from utils.replay import ReplayBuffer

import torch.distributions
import torch.nn as nn
import torch.nn.functional as F


# define network architecture
class Net(nn.Module):
    def __init__(self, num_obs=10, num_act=2):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, num_act),
        )

    def forward(self, x):
        return self.net(x)

# some useful functions
def soft_update(net, net_target, tau):
    # update target network with momentum (for approximating ground-truth Q-values)
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * tau + param.data * (1.0 - tau))


class DQN_basketball:
    def __init__(self, args):
        self.args = args

        # initialise parameters
        self.env = Basketball(args)
        self.replay = ReplayBuffer(buffer_limit= 5000, num_envs=args.num_envs)

        self.act_space = 2  # we discretise the action space into multiple bins (should be at least 2)
        self.discount = 0.99
        self.mini_batch_size = 128
        self.batch_size = self.args.num_envs * self.mini_batch_size
        self.tau = 0.995
        self.num_eval_freq = 100
        self.lr = 3e-4

        self.run_step = 1
        self.score = 0

        # define Q-network
        self.q        = Net(num_obs=10, num_act=self.act_space).to(self.args.sim_device)
        self.q_target = Net(num_obs=10, num_act=self.act_space).to(self.args.sim_device)
        soft_update(self.q, self.q_target, tau=0.0)
        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

    def update(self):
        # policy update using TD loss
        self.optimizer.zero_grad()

        obs, act, reward, next_obs, done_mask = self.replay.sample(self.mini_batch_size)
        q_table = self.q(obs)

        # map action from [-50, 50] to discrete index [0, act_space-1]
        act = torch.round((0.01 * (act + 50)) * (self.act_space - 1))

        # Q-value for each DOF
        q_val_0 = q_table[torch.arange(self.batch_size), act[:, 0].long()]
        q_val_1 = q_table[torch.arange(self.batch_size), act[:, 1].long()]

        # combine Q-values
        q_val = q_val_0 + q_val_1

        with torch.no_grad():
            q_val_next = self.q_target(next_obs).reshape(self.batch_size, -1).max(1)[0]

        target = reward + self.discount * q_val_next * done_mask

        loss = F.smooth_l1_loss(q_val, target)

        loss.backward()
        self.optimizer.step()

        # soft update target networks
        soft_update(self.q, self.q_target, self.tau)
        return loss

    def act(self, obs, epsilon=0.0):
        coin = torch.rand(self.args.num_envs, device=self.args.sim_device) < epsilon

        # random action in [-1, 1]
        rand_act = torch.rand(self.args.num_envs, 2, device=self.args.sim_device) * 2 - 1
        with torch.no_grad():
            q_table = self.q(obs)
            true_act = q_table

        # epsilon-greedy action selection
        act = coin.float().unsqueeze(-1) * rand_act + (1 - coin.float().unsqueeze(-1)) * true_act

        # map action to [-50, 50]
        act = act * 50
        act = torch.clamp(act, -50, 50)
        return act

    def run(self):
        epsilon = max(0.01, 0.8 - 0.01 * (self.run_step / 200))

        # collect data
        obs = self.env.obs_buf.clone()
        action = self.act(obs, epsilon)
        self.env.step(action)
        next_obs, reward, done = self.env.obs_buf.clone(), self.env.reward_buf.clone(), self.env.reset_buf.clone()
        self.env.reset()

        self.replay.push(obs, action, reward, next_obs, 1 - done)

        # training mode
        if self.replay.size() > self.mini_batch_size:
            loss = self.update()
            self.score += torch.mean(reward.float()).item() / self.num_eval_freq

            # evaluation mode
            if self.run_step % self.num_eval_freq == 0:
                print('Steps: {:04d} | Reward {:.04f} | TD Loss {:.04f} Epsilon {:.04f} Buffer {:03d}'
                      .format(self.run_step, self.score, loss.item(), epsilon, self.replay.size()))
                self.score = 0

        self.run_step += 1
