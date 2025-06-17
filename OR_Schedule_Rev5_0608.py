  tensorEquip = sample[:14]
  tensorQueue = sample
  J = tensorQueue.size(0) # number of jobs
  M = 14 # number of machines
  st = 0.16 # setup time
  
  tensorQueueCopy = copy.deepcopy(tensorQueue)
  tensorQueueCopy[:, -1][tensorQueue[:, -1] < 0] = tensorQueue[:, -1][tensorQueue[:, -1] > 0].min()
  p = tensorQueueCopy[:, -1].tolist()
  priority = tensorQueue[:, 9]
  BIG_M = J * (max(p) + st)
  # equip이 같지 않거나 0일때 false 그 외 True
  # matrixEquip = [[(torch.equal(equip[:2], job[:2])) and not (torch.equal(equip[:2], torch.tensor([0, 0]))) for job in tensorQueue] for equip in tensorEquip]
  matrixEquip = [[(torch.equal(equip[:2], job[:2])) for job in tensorQueue] for equip in tensorEquip]
  def pcardParser(tensorInput):
      if torch.equal(tensorInput, torch.tensor([1,0,0,0,0,0,1])): return 'FF44'
      elif torch.equal(tensorInput, torch.tensor([0,1,0,0,0,0,1])): return 'AV44'
      elif torch.equal(tensorInput, torch.tensor([0,0,0,1,0,1,0])): return 'JC30'
      else: return 'EMPT'
  matrixPcard = [pcardParser(job[2:9]) for job in tensorQueue]
  C = cp.Variable(J, nonneg=True)
  z = cp.Variable((J, M), boolean=True) # 할당 여부
  o = cp.Variable((J, J), boolean=True) # 순서 결정

  # def warm_start_heuristic(J, M, p, matrixEquip, matrixPcard, st=0.16):
  #     C_init = np.zeros(J)
  #     z_init = np.zeros((J, M))
  #     o_init = np.zeros((J, J))
  #     machine_ready = np.zeros(M)
  #     last_pcard = [""] * M

  #     # Step 1: Job 0~13 → Machine 0~13 고정 할당
  #     for j in range(M):
  #         z_init[j, j] = 1
  #         start_time = machine_ready[j]
  #         process_time = p[j]
  #         C_init[j] = start_time + process_time
  #         machine_ready[j] = C_init[j]
  #         last_pcard[j] = matrixPcard[j]
  #         # 순서정보: j보다 큰 job들에 대해 j가 먼저라고 설정
  #         for k in range(M, J):
  #             o_init[j, k] = 1

  #     # Step 2: 나머지 Job → 가능한 머신 중 가장 빨리 끝나는 곳에 할당
  #     for j in range(M, J):
  #         compatible_machines = [m for m in range(M) if matrixEquip[m][j]]
  #         if not compatible_machines:
  #             continue  # 할당 불가 → skip
  #         # setup time 고려한 실제 시작 가능 시간 계산
  #         earliest_times = [
  #             machine_ready[m] + (st if matrixPcard[j] != last_pcard[m] else 0)
  #             for m in compatible_machines
  #         ]
  #         m_best = compatible_machines[np.argmin(earliest_times)]
  #         start_time = earliest_times[np.argmin(earliest_times)]
  #         z_init[j, m_best] = 1
  #         C_init[j] = start_time + p[j]
  #         machine_ready[m_best] = C_init[j]
  #         last_pcard[m_best] = matrixPcard[j]

  #         # 순서 정보 설정
  #         for k in range(J):
  #             if k == j: continue
  #             if z_init[k, m_best] == 1 and C_init[k] <= start_time:
  #                 o_init[k, j] = 1

  #     return z_init, o_init, C_init

  # z_init, o_init, C_init = warm_start_heuristic(J, M, p, matrixEquip, matrixPcard, st=0.16)
  # z.value = z_init
  # o.value = o_init
  # C.value = C_init

  # ----- 제약 조건 정의 -----
  constraints = []
  
  # 각 Job은 정확히 하나의 Machine에 할당
  for j in range(J):
      constraints.append(cp.sum(z[j, :]) == 1)
      if j < M:
          constraints.append(z[j, j] == 1)
          constraints.append(C[j] == p[j])
          for k in range(M, J):
              constraints.append(o[j, k] == 1)
      else:
          constraints.append(C[j] >= p[j])

  # 장비 호환성
  for j in range(M, J):
      for m in range(M):
          if not matrixEquip[m][j]:
              constraints.append(z[j, m] == 0)  

  # 순서제약: 같은 machine에 할당된 job들만
  for i in range(J):
      for j in range(J):
          if i < j:
              constraints.append(o[i, j] + o[j, i] <= 1)
              if (i < M):
                  if (j >= M):
                      # i is machine
                      constraints.append(
                          C[j] >= C[i] + p[j] + st * (matrixPcard[i] != matrixPcard[j])
                          - BIG_M * (3 - z[i, i] - z[j, i] - o[i, j])
                      )
                      # constraints.append(o[i, j] + o[j, i] >= z[i, i] + z[j, i] - 1)
              else:
                  compatible_machines = [m for m in range(M) if matrixEquip[m][i] and matrixEquip[m][j]]
                  if not compatible_machines:
                      continue  # skip if no compatible machine pair
                  for m in compatible_machines:
                      constraints.append(
                          C[j] >= C[i] + p[j] + st * (matrixPcard[i] != matrixPcard[j])
                          - BIG_M * (3 - z[i, m] - z[j, m] - o[i, j])
                      )
                      constraints.append(
                          C[i] >= C[j] + p[i] + st * (matrixPcard[i] != matrixPcard[j])
                          - BIG_M * (3 - z[i, m] - z[j, m] - o[j, i])
                      )
                      
                      # 동일 machine에 2개의 job이 있을 경우 반드시 순서가 존재해야함
                      # constraints.append(z[i, m] + z[j, m] <= 1 + o[i, j] + o[j, i])
                      # if (i >= M) and (j >= M):
                      #     # 우선순위 반영 (높은 priority가 먼저)
                      #     if priority[i] < priority[j]:
                      #         constraints.append(o[j, i] >= z[i, m] + z[j, m] - 1)
                      #     else:
                      #         constraints.append(o[i, j] >= z[i, m] + z[j, m] - 1)
                      # else:
                      #     constraints.append(o[i, j] >= z[i, m] + z[j, m] - 1)
                      # constraints.append(o[i, j] + o[j, i] >= z[i, m] + z[j, m] - 1)

  priority_penalty_terms = []
  for i in range(J):
      for j in range(J):
          if i < j:
              if priority[i] < priority[j]:
                  priority_penalty_terms.append(1 - o[j, i])  # o[j, i]가 1이 아니면 패널티 발생
              else:
                  priority_penalty_terms.append(1 - o[i, j])
  # ----- 목적 함수: 평균 완료 시간 최소화 -----
  # obj = cp.Minimize(cp.mean(C))
  alpha = 5.0  # 우선순위 위반 패널티 가중치
  obj = cp.Minimize(
      cp.mean(C) + alpha * cp.sum(priority_penalty_terms)
  )

  # ----- 문제 정의 및 풀기 -----
  prob = cp.Problem(obj, constraints)
  # prob.solve(solver=cp.HIGHS, verbose=True, time_limit=600)
  prob.solve(solver=cp.HIGHS, verbose=True)

  # ----- 결과 출력 -----
  print('상태:', prob.status)
  print('평균 Start Time:', prob.value)
