
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
  # equip이 같지 않거나 0일때 false 그 외 True
  # matrixEquip = [[(torch.equal(equip[:2], job[:2])) and not (torch.equal(equip[:2], torch.tensor([0, 0]))) for job in tensorQueue] for equip in tensorEquip]
  matrixEquip = [[(torch.equal(equip[:2], job[:2])) for job in tensorQueue] for equip in tensorEquip]
  def pcardParser(tensorInput):
      if torch.equal(tensorInput, torch.tensor([1,0,0,0,0,0,1])): return 'FF44'
      elif torch.equal(tensorInput, torch.tensor([0,1,0,0,0,0,1])): return 'AV44'
      elif torch.equal(tensorInput, torch.tensor([0,0,0,1,0,1,0])): return 'JC30'
      else: return 'EMPT'
  matrixPcard = [pcardParser(job[2:9]) for job in tensorQueue]

  # Warm Start
  C_init = np.zeros(J)
  z_init = np.zeros((J, M))
  o_init = np.zeros((J, J))
  machineReady = np.zeros(M)
  for j in range(J):
      compatible = [m for m in range(M) if matrixEquip[m][j]]
      if not compatible:
          continue
      m = min(compatible, key=lambda m: machineReady[m])
      z_init[j, m] = 1
      startTime = machineReady[m]
      processTime = p[j]
      C_init[j] = startTime + processTime
      machineReady[m] = C_init[j] # + r
      for k in range(J):
          if k == j: continue
          if z_init[k, m] == 1 and C_init[k] <= startTime:
              o_init[k, j] = 1


  # HIGHS
  C = cp.Variable(J, nonneg=True)
  z = cp.Variable((J, M), boolean=True) # 할당 여부
  o = cp.Variable((J, J), boolean=True) # 순서 결정

  C.value = C_init
  z.value = z_init
  o.value = o_init

  # ----- 제약 조건 정의 -----
  constraints = []

  # 각 Job은 정확히 하나의 Machine에 할당
  for j in range(J):
      constraints.append(cp.sum(z[j, :]) == 1)
      constraints.append(C[j] >= p[j])

  # 장비 호환성
  for j in range(J):
      for m in range(M):
          if j == m:
              constraints.append(z[j, m] == 1)
          elif not matrixEquip[m][j]:
              constraints.append(z[j, m] == 0)  
              for k in range(M, J):
                  constraints.append(o[j, k] == 1)

  # 순서제약: 같은 machine에 할당된 job들만
  for i in range(J):
      for j in range(J):
          if i < j:
              constraints.append(o[i, j] + o[j, i] <= 1)
              if (i >= M) and (j >= M):
                  # 우선순위 반영 (높은 priority가 먼저)
                  if priority[i] < priority[j]:
                      constraints.append(o[j, i] == 1)
                  else:
                      constraints.append(o[i, j] == 1)
              for m in range(M):
                  constraints.append(
                      C[j] >= C[i] + p[j] + r * (matrixPcard[i] != matrixPcard[j])
                      - BIG_M * (3 - z[i, m] - z[j, m] - o[i, j])
                  )
                  constraints.append(
                      C[i] >= C[j] + p[i] + r * (matrixPcard[i] != matrixPcard[j])
                      - BIG_M * (3 - z[i, m] - z[j, m] - o[j, i])
                  )
                  # 동일 machine에 2개의 job이 있을 경우 반드시 순서가 존재해야함
                  constraints.append(z[i, m] + z[j, m] <= 1 + o[i, j] + o[j, i])
                  # constraints.append(o[i, j] >= z[i, m] + z[j, m] - 1)
              
  # 우선순위 반영 (높은 priority가 먼저)
  # for i in range(M, J):
  #     for j in range(M, J):
  #         if priority[i] > priority[j]:
  #             constraints.append(o[i, j] == 1)
  #         elif priority[i] < priority[j]:
  #             constraints.append(o[j, i] == 1)
  #         elif i < j:
  #             constraints.append(o[i, j] == 1)

  # ----- 목적 함수: 평균 시작 시간 최소화 -----
  idlePenalty = cp.sum([C[j+1] - (C[j] + p[j+1]) for j in range(J-1)])
  obj = cp.Minimize(cp.max(C) + 0.1*idlePenalty)

  # ----- 문제 정의 및 풀기 -----
  prob = cp.Problem(obj, constraints)
  prob.solve(solver=cp.HIGHS)

  # ----- 결과 출력 -----
  print('상태:', prob.status)
  print('평균 Start Time:', prob.value)
