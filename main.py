import torch
from torch.nn import Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

# 그래프 데이터 정의
def create_graph():
    # 노드 특징 (5차원, 각 작업/장비에 대한 임의의 초기 특성)
    # 작업: J1, J2, J3 / 장비: M1, M2
    node_features = torch.tensor([
        [1.0, 0.5, 0.0, 0.0, 0.0],  # J1
        [0.0, 1.0, 0.5, 0.0, 0.0],  # J2
        [0.0, 0.0, 1.0, 0.5, 0.0],  # J3
        [1.0, 1.0, 0.0, 0.0, 0.5],  # M1
        [0.5, 0.0, 1.0, 1.0, 0.0],  # M2
    ], dtype=torch.float32)

    # 엣지 정보 (edge_index)
    # 작업 간 선행 조건: J1 -> J2, J2 -> J3
    # 작업-장비 연결: J1 <-> M1, J2 <-> M2, J3 <-> M1
    edge_index = torch.tensor([
        [0, 1, 1, 2, 0, 3, 3, 1, 4, 4, 2, 3],  # Source nodes
        [1, 2, 0, 1, 3, 0, 1, 4, 1, 2, 3, 2],  # Target nodes
    ], dtype=torch.long)

    # 그래프 데이터 생성
    graph_data = Data(x=node_features, edge_index=edge_index)
    return graph_data

# GNN 모델 정의
class JobShopGraphConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(JobShopGraphConv, self).__init__()
        # GraphConv 레이어 정의
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        # Fully Connected Layer 정의
        self.fc = Linear(hidden_dim, output_dim)
        self.relu = ReLU()

    def forward(self, x, edge_index):
        # GraphConv 레이어 1
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        # GraphConv 레이어 2
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # 최종 출력
        x = self.fc(x)
        return x


# 그래프 데이터 생성
graph_data = create_graph()

# 모델 정의
input_dim = graph_data.x.shape[1]  # 입력 차원 (노드 특성 차원)
hidden_dim = 16                    # 히든 레이어 차원
output_dim = 1                     # 출력 차원 (예: 스케줄링 점수)
model = JobShopGraphConv(input_dim, hidden_dim, output_dim)

# 손실 함수 및 최적화기 정의
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습 데이터 (예: 노드별 스케줄링 점수 목표)
y_true = torch.tensor([[0.8], [0.5], [0.3], [0.7], [0.6]], dtype=torch.float32)

# 학습 루프
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # Forward Pass
    output = model(graph_data.x, graph_data.edge_index)

    # 손실 계산
    loss = criterion(output, y_true)
    loss.backward()
    optimizer.step()

    # Loss 출력
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 추론
model.eval()
with torch.no_grad():
    predictions = model(graph_data.x, graph_data.edge_index)
    print("Predicted Scheduling Scores:")
    print(predictions)
