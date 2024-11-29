import torch
from torch_geometric.data import Data

# 입력 그래프 생성
def create_initial_graph(num_jobs, num_machines):
    # 노드 특징 (임의의 초기 특성 정의)
    node_features = torch.rand((num_jobs + num_machines, 5))  # 예: 5차원 노드 특성

    # 작업 간 선행 조건 (작업 간 연결만 포함)
    task_precedence_edges = torch.tensor([
        [0, 1],  # J1 → J2
        [1, 2]   # J2 → J3
    ], dtype=torch.long).t()

    # 작업-장비 간 연결은 초기 상태에서는 포함하지 않음
    return Data(x=node_features, edge_index=task_precedence_edges)

# 예제: 3개의 작업, 2개의 장비
graph_data = create_initial_graph(num_jobs=3, num_machines=2)

import torch.nn as nn
from torch_geometric.nn import GraphConv

class JobSchedulerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_machines):
        super(JobSchedulerGNN, self).__init__()
        # GNN Layers
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        # Fully Connected Layers for Task-Machine Allocation
        self.fc_allocation = nn.Linear(hidden_dim, num_machines)

        # Fully Connected Layers for Task Order Prediction
        self.fc_order = nn.Linear(hidden_dim, 1)  # 순서 점수 출력

    def forward(self, x, edge_index):
        # GNN 레이어를 통한 노드 임베딩 생성
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # 작업-장비 할당 예측
        task_allocation = self.fc_allocation(x)

        # 작업 간 순서 예측
        task_order = self.fc_order(x)

        return task_allocation, task_order

# 데이터와 모델 준비
num_jobs = 3
num_machines = 2
graph_data = create_initial_graph(num_jobs, num_machines)
model = JobSchedulerGNN(input_dim=graph_data.x.shape[1], hidden_dim=16, num_machines=num_machines)

# 손실 함수 및 옵티마이저
allocation_criterion = nn.CrossEntropyLoss()  # 작업-장비 할당용
order_criterion = nn.MSELoss()  # 작업 간 순서 점수용
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습 데이터
allocation_labels = torch.tensor([0, 1, 0])  # J1 -> M1, J2 -> M2, J3 -> M1
order_labels = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)  # 작업 순서 점수

# 학습 루프
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    allocation_output, order_output = model(graph_data.x, graph_data.edge_index)

    # 작업-장비 할당 손실 계산 (작업 노드만 사용)
    task_allocation_loss = allocation_criterion(allocation_output[:num_jobs], allocation_labels)

    # 작업 간 순서 손실 계산 (작업 노드만 사용)
    task_order_loss = order_criterion(order_output[:num_jobs].squeeze(), order_labels)

    # 총 손실 및 역전파
    total_loss = task_allocation_loss + task_order_loss
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")

# 추론
model.eval()
with torch.no_grad():
    allocation_output, order_output = model(graph_data.x, graph_data.edge_index)

    # 작업-장비 할당 결과
    task_to_machine = torch.argmax(allocation_output[:num_jobs], dim=1).tolist()
    print(f"Task to Machine Allocation: {task_to_machine}")

    # 작업 간 순서 결과
    task_order = order_output[:num_jobs].squeeze().tolist()
    print(f"Task Order Scores: {task_order}")
