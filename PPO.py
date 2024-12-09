import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch.distributions import Categorical

# GNN 기반 정책 네트워크 정의
class GNNPolicy(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, action_dim):
        super(GNNPolicy, self).__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)  # 행동 선택
        self.critic = nn.Linear(hidden_dim, 1)         # 상태 가치 함수

    def forward(self, x, edge_index):
        # GNN을 통해 상태 임베딩 계산
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x

    def get_action(self, x, edge_index):
        # 행동 선택
        x = self.forward(x, edge_index)
        logits = self.actor(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_value(self, x, edge_index):
        # 상태 가치 평가
        x = self.forward(x, edge_index)
        return self.critic(x)

# 강화학습 환경 정의 (예제)
class SchedulingEnv:
    def __init__(self, num_jobs, num_machines):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.state = None

    def reset(self):
        # 초기 상태 설정
        self.state = torch.zeros(self.num_jobs + self.num_machines, 10)  # 예: 임베딩 초기화
        return self.state

    def step(self, action):
        # 행동(action)을 적용한 후 보상 계산
        reward = -1  # 보상 예시 (작업 완료 시간 최소화)
        done = False  # 에피소드 종료 여부
        return self.state, reward, done

# PPO 학습 과정
class PPO:
    def __init__(self, policy, optimizer, gamma=0.99, clip_eps=0.2, update_iters=10):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_iters = update_iters

    def compute_advantages(self, rewards, values):
        advantages = []
        gae = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            gae = r + self.gamma * gae - v
            advantages.insert(0, gae)
        return torch.tensor(advantages)

    def update(self, states, actions, log_probs, rewards, next_states, edge_index):
        # PPO 업데이트
        values = self.policy.get_value(states, edge_index)
        next_values = self.policy.get_value(next_states, edge_index)
        advantages = self.compute_advantages(rewards, values)
        for _ in range(self.update_iters):
            new_log_probs = []
            for state in states:
                _, log_prob = self.policy.get_action(state, edge_index)
                new_log_probs.append(log_prob)
            new_log_probs = torch.stack(new_log_probs)

            # PPO 손실 계산
            ratio = (new_log_probs - log_probs).exp()
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = (rewards - values).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 주요 학습 루프
def train():
    env = SchedulingEnv(num_jobs=10, num_machines=5)
    policy = GNNPolicy(node_dim=10, edge_dim=5, hidden_dim=64, action_dim=5)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    ppo = PPO(policy, optimizer)

    for episode in range(1000):
        state = env.reset()
        done = False
        states, actions, rewards, log_probs = [], [], [], []

        while not done:
            edge_index = torch.tensor([[0, 1], [1, 0]])  # 예제 간선 정의
            action, log_prob = policy.get_action(state, edge_index)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        ppo.update(states, actions, log_probs, rewards, next_state, edge_index)

train()