from ortools.sat.python import cp_model

# 기본 설정
J = 20  # Jobs
M = 5   # Machines
st = 1  # Setup time

import numpy as np
np.random.seed(42)
p = np.random.randint(5, 20, size=J)
priority = np.random.randint(1, 5, size=J)
matrixEquip = np.random.choice([True, False], size=(M, J), p=[0.7, 0.3])
matrixPcard = np.random.choice(['A', 'B'], size=J)

model = cp_model.CpModel()
BIG_M = sum(p) * 2

# 변수
C = [model.NewIntVar(0, BIG_M, f'C_{j}') for j in range(J)]
z = [[model.NewBoolVar(f'z_{j}_{m}') for m in range(M)] for j in range(J)]
o = [[model.NewBoolVar(f'o_{i}_{j}') for j in range(J)] for i in range(J)]
max_C = model.NewIntVar(0, BIG_M, 'max_C')

# 제약조건
for j in range(J):
    model.Add(sum(z[j][m] for m in range(M)) == 1)
    model.Add(C[j] >= p[j])
    for m in range(M):
        if not matrixEquip[m][j]:
            model.Add(z[j][m] == 0)

for i in range(J):
    for j in range(J):
        if i == j:
            continue
        for m in range(M):
            setup = st if matrixPcard[i] != matrixPcard[j] else 0
            model.Add(C[j] >= C[i] + p[j] + setup - (1 - z[i][m]) * BIG_M - (1 - z[j][m]) * BIG_M - (1 - o[i][j]) * BIG_M)
            model.Add(C[i] >= C[j] + p[i] + setup - (1 - z[i][m]) * BIG_M - (1 - z[j][m]) * BIG_M - (1 - o[j][i]) * BIG_M)
            model.Add(o[i][j] + o[j][i] >= z[i][m] + z[j][m] - 1)

for i in range(J):
    for j in range(J):
        if i != j and priority[i] < priority[j]:
            model.Add(o[i][j] == 1)

model.AddMaxEquality(max_C, C)
model.Minimize(max_C)

# Solver
solver = cp_model.CpSolver()
status = solver.Solve(model)

# 결과 출력
if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    for j in range(J):
        m_assigned = [m for m in range(M) if solver.Value(z[j][m]) == 1][0]
        print(f"Job {j}: Machine {m_assigned}, Completion Time = {solver.Value(C[j])}, Priority = {priority[j]}")
else:
    print("No solution found.")