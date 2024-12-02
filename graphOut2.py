import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# 그래프 데이터 준비
def create_graph(job_nodes, machine_nodes, edges):
    """
    job_nodes: Job Node 수
    machine_nodes: Machine Node 수
    edges: 초기 연결 정보 [(job_idx, machine_idx), ...]
    """
    total_nodes = job_nodes + machine_nodes

    # 노드의 feature 정의 (예: 임의 feature)
    x = torch.rand(total_nodes, 16)  # 16차원 feature
    
    # 엣지 정의
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Job 노드와 장비 노드의 레이블
    node_labels = torch.cat([torch.zeros(job_nodes), torch.ones(machine_nodes)])
    
    return Data(x=x, edge_index=edge_index, y=node_labels)

# 데이터 예시
job_nodes = 10
machine_nodes = 5
edges = [(0, 10), (1, 10), (2, 11), (3, 12)]  # 일부 Job -> Machine 연결

data = create_graph(job_nodes, machine_nodes, edges)

# GNN 모델 정의
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 모델 초기화
input_dim = 16
hidden_dim = 32
output_dim = 16
model = GNNModel(input_dim, hidden_dim, output_dim)

# 손실 함수 및 최적화
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습 루프
def train(data, model, epochs=200):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # 연결 예측을 위한 손실 정의
        loss = F.mse_loss(out[data.edge_index[0]], out[data.edge_index[1]])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# 모델 학습
train(data, model)

# 미연결 Job -> Machine 예측
def predict(data, model, job_idx, machine_idx):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        job_feature = out[job_idx]
        machine_feature = out[machine_idx]
        similarity = F.cosine_similarity(job_feature, machine_feature)
        return similarity

# 예시: Job 4와 모든 Machine 노드 간의 연결 점수
job_idx = 4
for machine_idx in range(job_nodes, job_nodes + machine_nodes):
    score = predict(data, model, job_idx, machine_idx)
    print(f"Job {job_idx} -> Machine {machine_idx}: Score = {score.item():.4f}")