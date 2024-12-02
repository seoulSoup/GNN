import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt

# 1. 입력 그래프 정의
# 노드 특징 벡터 (x), 엣지 연결 정보 (edge_index)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 0],  # 출발 노드
    [1, 0, 2, 1, 3, 2, 0, 3]   # 도착 노드
], dtype=torch.long)

node_features = torch.tensor([
    [1.0], [2.0], [3.0], [4.0]
], dtype=torch.float)  # 노드 특징 (예: 각 노드의 상태)

data = Data(x=node_features, edge_index=edge_index)

# 2. GNN 모델 정의
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 첫 번째 GCN 레이어
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 두 번째 GCN 레이어
        x = self.conv2(x, edge_index)
        return x

# 3. 모델 초기화 및 학습 (여기서는 간단히 Forward Pass만 수행)
model = GNNModel(in_channels=1, hidden_channels=4, out_channels=2)
output = model(data)  # 노드 예측값 (노드당 2차원 출력)

# 4. 엣지 수준 예측 (optional)
# GNN으로 엣지 수준의 값도 예측 가능. 여기서는 엣지 가중치를 예측한다고 가정.
edge_weights = torch.sigmoid(torch.rand(data.edge_index.size(1)))  # 랜덤 예측 (0~1)

# 5. 출력 그래프 생성
# NetworkX 그래프로 변환
output_graph = nx.Graph()
output_graph.add_edges_from(data.edge_index.t().tolist())

# 노드 예측 결과 추가 (예: 클래스 또는 특성 벡터)
predicted_node_labels = {i: output[i].argmax().item() for i in range(data.x.size(0))}
nx.set_node_attributes(output_graph, predicted_node_labels, "predicted_label")

# 엣지 예측 결과 추가 (예: 가중치)
predicted_edge_weights = {
    (int(data.edge_index[0, i]), int(data.edge_index[1, i])): float(edge_weights[i])
    for i in range(data.edge_index.size(1))
}
nx.set_edge_attributes(output_graph, predicted_edge_weights, "predicted_weight")

# 6. 출력 그래프 확인
print("Nodes with attributes:", output_graph.nodes(data=True))
print("Edges with attributes:", output_graph.edges(data=True))

# 7. 그래프 시각화
node_colors = [output_graph.nodes[n]["predicted_label"] for n in output_graph.nodes]
edge_weights_list = [output_graph.edges[e]["predicted_weight"] for e in output_graph.edges]

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(output_graph)
nx.draw(output_graph, pos, with_labels=True, node_color=node_colors, cmap='viridis', node_size=500)
nx.draw_networkx_edge_labels(output_graph, pos, edge_labels=predicted_edge_weights)
plt.show()