for sample in listDataSet:
    tensorEquip = sample[:14]
    tensorQueue = sample
    J = tensorQueue.size(0) # number of jobs
    M = 14 # number of machines
    r = 0.16 # release time
    BIG_M = 1000
    tensorQueueCopy = copy.deepcopy(tensorQueue)
    tensorQueueCopy[:, -1][tensorQueue[:, -1] < 0] = tensorQueue[:, -1][tensorQueue[:, -1] > 0].min()
    p = tensorQueueCopy[:, -1].tolist()
    priority = tensorQueue[:, 9]
    matrixEquip = [[torch.equal(equip[:2], job[:2]) for job in tensorQueue] for equip in tensorEquip]
    def pcardParser(tensorInput):
        if torch.equal(tensorInput, torch.tensor([1,0,0,0,0,0,1])): return 'FF44'
        elif torch.equal(tensorInput, torch.tensor([0,1,0,0,0,0,1])): return 'AV44'
        elif torch.equal(tensorInput, torch.tensor([0,0,0,1,0,1,0])): return 'JC30'
        else: return 'EMPT'
    matrixPcard = [pcardParser(job[2:9]) for job in tensorQueue]
    
    # s = cp.Variable(J, nonneg=True) # 시작 시간
    C = cp.Variable(J, nonneg=True) # 완료 시간
    z = cp.Variable((J, M), boolean=True) # 할당 여부
    o = cp.Variable((J, J), boolean=True) # 순서 결정

    constraints = []

    # 0. 초기 조건 (이미 할당된 Plan 들을 Job의 M번째까지 초기 조건으로 할당)
    for j in range(J):
        for m in range(M):
            if not matrixEquip[m][j]:
                constraints.append(z[j, m] == 0)     
            elif j < M:
                if j == m:
                    constraints.append(z[j, m] == 1)
                for k in range(M, J):
                    constraints.append(C[j] <= C[k] + BIG_M * (1 - z[k, m]))
    

    # 1. 각 Job은 하나의 Machine에만 할당
    for j in range(J):
        constraints.append(cp.sum(z[j, :]) == 1)
        # 2. 완료 시간 정의
        constraints.append(C[j] >= p[j])

    # 2. Priority 조건
    for j in range(J):
        for k in range(J):
            if priority[j] < priority[k]:
                constraints.append(o[j, k] == 0)
                
    for j in range(J):
        for k in range(J):
            if j != k:
                for m in range(M):
                    sameMachine = z[j, m] + z[k, m] - 1 # m에 j,k 둘 다 할당 되었으면 1임
                    partsDifferent = int(matrixPcard[j] != matrixPcard[k])
                    constraints.append(C[j] <= C[k] - (p[k] + r*partsDifferent)*(1 - o[j, k] + 1 - sameMachine) + BIG_M*(1 - o[j, k]))
                    # constraints.append(C[k] <= C[j] - p[j]*(o[j, k] + 1 - sameMachine)) - r*partsDifferent + BIG_M*(o[j, k])
                    constraints.append(o[j, k] + o[k, j] == 1)


    obj = cp.Minimize(cp.max(C))
    # raise ValueError
    prob = cp.Problem(obj, constraints)
    # prob.solve(solver='CBC')
    prob.solve(solver=cp.HIGHS)
    print('최적해 상태: ', prob.status)
    print('최적값: ', prob.value)

    break
    
import plotly.express as px
import pandas as pd

ganttData = []
for j in range(J):
    idMachine = int(np.argmax(z.value[j], axis=0))
    ganttData.append({
        'Job': f'Job {j}',
        'Start': float(s.value[j]),
        'Finish': float(C.value[j]),
        'Machine': f'Mahchine {idMachine}'
    })
dfGantt = pd.DataFrame(ganttData)
print(dfGantt)

fig = px.timeline(dfGantt, 
x_start='Start', 
x_end='Finish', 
y='Machine', 
color='Job', 
title='Gantt Chart',
hover_data=['Job'])
fig.update_yaxes(title='Machine Index', autorange='reversed', categoryorder='category ascending')
fig.show()
