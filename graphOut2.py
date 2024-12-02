import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# 모델, 데이터는 이전 코드와 동일
# ...

# 미연결 Job -> Machine 예측 (효율적 방식)
def predict_all_connections(data, model, job_count, machine_count):
    model.eval()
    with torch.no_grad():
        # 모든 노드의 임베딩 계산
        out = model(data.x, data.edge_index)
        
        # Job 노드와 Machine 노드 임베딩 분리
        job_embeddings = out[:job_count]  # Job 노드 임베딩
        machine_embeddings = out[job_count:job_count + machine_count]  # Machine 노드 임베딩
        
        # 코사인 유사도 계산
        # job_embeddings: (J, D), machine_embeddings: (M, D)
        # similarity_matrix: (J, M)
        similarity_matrix = torch.matmul(job_embeddings, machine_embeddings.t()) / (
            torch.norm(job_embeddings, dim=1, keepdim=True) *
            torch.norm(machine_embeddings, dim=1, keepdim=True).t()
        )
        
        return similarity_matrix

# 예시: 모든 Job -> Machine 간 유사도 계산
job_count = 10
machine_count = 5
similarity_matrix = predict_all_connections(data, model, job_count, machine_count)

# 유사도 행렬 출력
print("Similarity Matrix (Job -> Machine):")
print(similarity_matrix)

# 각 Job이 가장 높은 유사도를 가진 Machine에 연결
top_machine_for_each_job = similarity_matrix.argmax(dim=1)
for job_idx, machine_idx in enumerate(top_machine_for_each_job):
    print(f"Job {job_idx} -> Machine {machine_idx.item() + job_count}")