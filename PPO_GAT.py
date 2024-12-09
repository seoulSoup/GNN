import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.data import Data

# GAT 기반 정책 네트워크 정의
class GATPolicy(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_actions):
        super(GATPolicy, self).__init__()
        self.gat1 = GATConv(node_dim, hidden_dim, heads=1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.actor = nn.Linear(hidden_dim, num_actions)  # 행동 확률 계산
        self.critic = nn.Linear(hidden_dim, 1)          # 상태 가치 계산

    def forward(self, x, edge_index):
        # GAT 레이어를 통해 노드 임베딩 계산
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        return x

    def get_action(self, x, edge_index, mask=None):
        x = self.forward(x, edge_index)
        logits = self.actor(x)
        if mask is not None:
            logits = logits + (mask - 1) * 1e9  # Masked Softmax
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def get_value(self, x, edge_index):
        x = self.forward(x, edge_index)
        return self.critic(x)

# PPO 알고리즘 정의
class PPO:
    def __init__(self, policy, optimizer, gamma=0.99, clip_eps=0.2, update_iters=10):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_iters = update_iters

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae * (1 - dones[t])
            advantages.insert(0, gae)
        return torch.tensor(advantages)

    def update(self, states, actions, log_probs, rewards, next_states, edge_index, dones):
        values = torch.cat([self.policy.get_value(state, edge_index) for state in states])
        next_values = torch.cat([self.policy.get_value(next_state, edge_index) for next_state in next_states])
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_iters):
            new_log_probs = []
            entropy = []
            for state in states:
                _, log_prob, ent = self.policy.get_action(state, edge_index)
                new_log_probs.append(log_prob)
                entropy.append(ent)
            new_log_probs = torch.stack(new_log_probs)
            entropy = torch.stack(entropy).mean()

            ratio = (new_log_probs - log_probs).exp()
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = F.mse_loss(rewards, values)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 환경 정의 (예제)
class SchedulingEnv:
    def __init__(self, num_jobs, num_machines):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.state = None

    def reset(self):
        # 초기 그래프 상태 설정
        num_nodes = self.num_jobs + self.num_machines
        self.state = torch.rand(num_nodes, 8)  # 예제 노드 특성 (차원: 8)
        return self.state

    def step(self, action):
        # 행동(action)을 적용한 후 보상 계산
        reward = -torch.rand(1).item()  # 임의의 보상
        done = torch.rand(1).item() > 0.95  # 임의의 종료 조건
        next_state = self.state  # 상태는 변경되지 않음 (예제)
        return next_state, reward, done

# 학습 루프
def train():
    env = SchedulingEnv(num_jobs=5, num_machines=3)
    policy = GATPolicy(node_dim=8, hidden_dim=16, num_actions=3)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    ppo = PPO(policy, optimizer)

    for episode in range(1000):
        state = env.reset()
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 예제 간선 정보
        done = False
        states, actions, rewards, log_probs, next_states, dones = [], [], [], [], [], []

        while not done:
            action, log_prob, _ = policy.get_action(state, edge_index)
            next_state, reward, done = env.step(action)

            # 데이터 저장
            states.append(state)
            actions.append(action)
            rewards.append(torch.tensor(reward))
            log_probs.append(log_prob)
            next_states.append(next_state)
            dones.append(torch.tensor(done, dtype=torch.float))

            state = next_state

        # PPO 업데이트
        ppo.update(states, actions, log_probs, rewards, next_states, edge_index, dones)

train()