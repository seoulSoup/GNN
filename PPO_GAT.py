import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.data import Data

# GAT 기반 정책 네트워크 정의
class GATPolicy(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super(GATPolicy, self).__init__()
        self.gat1 = GATConv(node_dim, hidden_dim, heads=1)
        self.gat2 = GATConv(hidden_dim, 1, heads=1)  # Attention weights 출력
        self.critic = nn.Linear(hidden_dim, 1)      # Critic 네트워크

    def forward(self, x, edge_index):
        # GAT을 통해 attention weights 계산
        x = F.relu(self.gat1(x, edge_index))
        attention_weights = self.gat2(x, edge_index).squeeze(-1)  # [num_edges]
        return x, attention_weights

    def get_action(self, x, edge_index):
        # Attention 값을 기반으로 행동 선택
        _, attention_weights = self.forward(x, edge_index)
        edge_attention = torch.softmax(attention_weights, dim=0)  # Normalize for numerical stability
        selected_edge = torch.argmax(edge_attention)  # 최대 attention을 가진 간선 선택
        return selected_edge, attention_weights

    def get_value(self, x, edge_index):
        x, _ = self.forward(x, edge_index)
        return self.critic(x.mean(dim=0))  # 상태 가치 계산

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

    def update(self, states, actions, rewards, next_states, edge_index, dones):
        values = torch.cat([self.policy.get_value(state, edge_index) for state in states])
        next_values = torch.cat([self.policy.get_value(next_state, edge_index) for next_state in next_states])
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_iters):
            new_values = torch.cat([self.policy.get_value(state, edge_index) for state in states])
            actor_loss = 0
            for state, action in zip(states, actions):
                _, attention_weights = self.policy.get_action(state, edge_index)
                log_prob = torch.log(attention_weights[action])  # Action log probability
                ratio = torch.exp(log_prob - log_prob.detach())
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                actor_loss += -torch.min(surrogate1, surrogate2).mean()

            critic_loss = F.mse_loss(rewards, values)
            loss = actor_loss + 0.5 * critic_loss

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
        num_nodes = self.num_jobs + self.num_machines
        self.state = torch.rand(num_nodes, 8)  # 예제 노드 특성
        return self.state

    def step(self, action):
        reward = -torch.rand(1).item()  # 임의의 보상
        done = torch.rand(1).item() > 0.95  # 임의의 종료 조건
        next_state = self.state  # 상태는 변경되지 않음 (예제)
        return next_state, reward, done

# 학습 루프
def train():
    env = SchedulingEnv(num_jobs=5, num_machines=3)
    policy = GATPolicy(node_dim=8, hidden_dim=16)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    ppo = PPO(policy, optimizer)

    for episode in range(1000):
        state = env.reset()
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 예제 간선 정보
        done = False
        states, actions, rewards, next_states, dones = [], [], [], [], []

        while not done:
            action, attention_weights = policy.get_action(state, edge_index)
            next_state, reward, done = env.step(action)

            # 데이터 저장
            states.append(state)
            actions.append(action)
            rewards.append(torch.tensor(reward))
            next_states.append(next_state)
            dones.append(torch.tensor(done, dtype=torch.float))

            state = next_state

        # PPO 업데이트
        ppo.update(states, actions, rewards, next_states, edge_index, dones)

train()