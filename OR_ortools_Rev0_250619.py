tensorEquip = sample[:14]
tensorQueue = sample[:40] if sample.size(0) > 40 else sample
J = tensorQueue.size(0) # number of jobs
M = 14 # number of machines
st = 0.16 # setup time

tensorQueueCopy = copy.deepcopy(tensorQueue)
tensorQueueCopy[:, -1][tensorQueue[:, -1] < 0] = tensorQueue[:, -1][tensorQueue[:, -1] > 0].min()
SCALE = 10
p = [int(v*SCALE) for v in tensorQueueCopy[:, -1].tolist()]
priority = tensorQueue[:, 9]
# BIG_M = J * (max(p) + st)
BIG_M = sum(p) + SCALE
print(f'BIG_M: {BIG_M}')
# equip이 같지 않거나 0일때 false 그 외 True
# matrixEquip = [[(torch.equal(equip[:2], job[:2])) and not (torch.equal(equip[:2], torch.tensor([0, 0]))) for job in tensorQueue] for equip in tensorEquip]
# matrixEquip = [[True if torch.equal(equip[:2], torch.tensor([1, 0])) else (torch.equal(equip[:2], job[:2])) for job in tensorQueue] for equip in tensorEquip]
matrixEquip = []
for equip in tensorEquip:
    listTemp = []
    for job in tensorQueue:
        if torch.equal(equip[:2], torch.tensor([0, 0])):
            listTemp.append(False)
        elif torch.equal(equip[:2], torch.tensor([1, 0])):
            listTemp.append(True)
        else:
            listTemp.append(torch.equal(equip[:2], job[:2]))
    matrixEquip.append(listTemp)

def pcardParser(tensorInput):
    if torch.equal(tensorInput, torch.tensor([1,0,0,0,0,0,1])): return 'FF44'
    elif torch.equal(tensorInput, torch.tensor([0,1,0,0,0,0,1])): return 'AV44'
    elif torch.equal(tensorInput, torch.tensor([0,0,0,1,0,1,0])): return 'JC30'
    else: return 'EMPT'
matrixPcard = [pcardParser(job[2:9]) for job in tensorQueue]

model = cp_model.CpModel()
C = [model.NewIntVar(0, BIG_M, f'C_{j}') for j in range(J)]
z = [[model.NewBoolVar(f'z_{j}_{m}') for m in range(M)] for j in range(J)]
o = [[model.NewBoolVar(f'o_{i}_{j}') for j in range(J)] for i in range(J)]
# o = [[None for j in range(J)] for i in range(J)]

max_C = model.NewIntVar(0, BIG_M, 'max_C')
print('Before Constraints')
###### 제약조건 ######
# 초기 조건
for j in range(J):
    model.Add(sum(z[j][m] for m in range(M)) == 1)
    model.Add(C[j] >= p[j])
    if j < M:
        model.Add(z[j][j] == 1)
        model.Add(C[j] == p[j])
        for k in range(M, J):
            model.Add(o[j][k] == 1)
            # o[j][k] = model.NewBoolVar(f'o_{j}_{k}')

# equip 일치 안하면 배정 안함
for j in range(M, J):
    for m in range(M):
        if not matrixEquip[m][j]:
            model.Add(z[j][m] == 0)

for i in range(J):
    for j in range(M, J):
        if i < j:
            # o[i][j] = model.NewBoolVar(f'o_{i}_{j}')
            # o[j][i] = model.NewBoolVar(f'o_{j}_{i}')
            model.Add(o[i][j] + o[j][i] <= 1)
            if (i < M):
                # i is machine
                model.Add(C[j] >= C[i] + p[j] + int(st*SCALE) * (matrixPcard[i] != matrixPcard[j]) - (3 - z[i][m] - z[j][m] - o[i][j]) * BIG_M)
                # model.Add(o[i][j] + o[j][i] >= z[i][m] + z[j][m] - 1)
            else:
                compatible_machines = [m for m in range(M) if matrixEquip[m][i] and matrixEquip[m][j]]
                if not compatible_machines:
                    continue  # skip if no compatible machine pair
                for m in compatible_machines:
                    model.Add(C[j] >= C[i] + p[j] + int(st*SCALE) * (matrixPcard[i] != matrixPcard[j]) - (3 - z[i][m] - z[j][m] - o[i][j]) * BIG_M)
                    model.Add(C[i] >= C[j] + p[i] + int(st*SCALE) * (matrixPcard[i] != matrixPcard[j]) - (3 - z[i][m] - z[j][m] - o[j][i]) * BIG_M)
                    # model.Add(o[i][j] + o[j][i] >= z[i][m] + z[j][m] - 1)
                
                # priority
                # if priority[i] < priority[j]:
                #     model.Add(o[j][i] == 1)
                # else:
                #     model.Add(o[i][j] == 1)

model.AddMaxEquality(max_C, C)
model.Minimize(max_C)
# model.Minimize(sum(C))

proto = model.Proto()
print(f'Number of Variables: {len(proto.variables)}')
print(f'Number of constraints: {len(proto.constraints)}')

print(f'Proto total size approx: ', sys.getsizeof(proto))
# Solver
solver = cp_model.CpSolver()
solver.parameters.max_memory_in_mb = 2048
solver.parameters.num_search_workers = 1
solver.parameters.max_time_in_seconds = 300
solver.parameters.log_search_progress = True

status = solver.Solve(model)

# 결과 출력
if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    for j in range(J):
        m_assigned = [m for m in range(M) if solver.Value(z[j][m]) == 1][0]
        print(f"Job {j}: Machine {m_assigned}, Completion Time = {solver.Value(C[j])}, Priority = {priority[j]}")
else:
    print("No solution found.")
        
