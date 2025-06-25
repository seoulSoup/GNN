  def pcardParser(tensorInput):
      if torch.equal(tensorInput, torch.tensor([1,0,0,0,0,0,1])): return 'FF44'
      elif torch.equal(tensorInput, torch.tensor([0,1,0,0,0,0,1])): return 'AV44'
      elif torch.equal(tensorInput, torch.tensor([0,0,0,1,0,1,0])): return 'JC30'
      else: return 'EMPT'

  # ----------------------- 입력 데이터 -----------------------
  M = 14                     # machine 개수
  tensorQueue = sample[:50]  # 최대 50개만 사용
  # tensorQueue = sample
  J = tensorQueue.size(0)
  print(f'Jobs: {J}')
  SCALE      = 10                            # 0.1 단위
  dur_float  = tensorQueue[:, -1].numpy()    # 처리시간(float)
  p          = [int((max(v, 0.1) if v > 0 else 0) * SCALE) for v in dur_float]  # 0 이하면 최소 0.1
  

  priority   = tensorQueue[:, 9].numpy()
  
  setup_val  = int(0.2 * SCALE)                              # setup time 정수

  # 장비 호환성: matrixEquip[m][j] == True 면 m에서 j 수행 가능
  matrixEquip = []
  for equip in sample[:M]:
      row = []
      for job in tensorQueue:
          if torch.equal(equip[:2], torch.tensor([0, 0])):
              row.append(False)
          elif torch.equal(equip[:2], torch.tensor([1, 0])):  # wildcard 장비
              row.append(True)
          else:
              row.append(torch.equal(equip[:2], job[:2]))
      matrixEquip.append(row)
  # ----------------------- CP-SAT 모델 -----------------------
  model = cp_model.CpModel()
  H     = sum(p) + setup_val * J                  # 수평선(upper bound)
  print(f'Horizon: {H}')
  pcard      = [pcardParser(t[2:9]) for t in tensorQueue]     # 'FF44' 등
  matrixPcard      = [[pcard[r] != pcard[c] if r < c else False for c in range(J)] for r in range(J)]     # 'FF44' 등
  # matrixPcard      = [[pcard[r] != pcard[c] for c in range(J)] for r in range(J)]     # 'FF44' 등
  # matrixSeq      = [[model.NewBoolVar(f'seq_{r}_{c}') for c in range(J)] for r in range(J)]     # 'FF44' 등
  # pConstant  = [model.NewConstant(int((max(v, 0.1) if v > 0 else 0) * SCALE)) for v in dur_float]  # 0 이하면 최소 0.1
  # Job 시간 변수
  start = [model.NewIntVar(0, H, f'start_{j}') for j in range(J)]
  end   = [model.NewIntVar(0, H, f'end_{j}')   for j in range(J)]

  # 머신 할당 bool + interval
  is_on = [[None]*M for _ in range(J)] # z
  intervals = [[] for _ in range(M)]
  

  # ------------- 각 작업의 Interval Variable 생성 ----------------
  for j in range(J):
      for m in range(M):
          if (matrixEquip[m][j]) or (j < M):
              is_on[j][m] = model.NewBoolVar(f'is_j{j}_m{m}')
              iv = model.NewOptionalIntervalVar(
                      start[j], p[j], end[j], is_on[j][m], f'intv_j{j}_m{m}')
              intervals[m].append(iv)
          else:
              is_on[j][m] = model.NewConstant(0)
      # 각 job은 하나의 machine에만 할당
      model.AddExactlyOne(is_on[j])
  
  # ------------- 초기 Job 0~M-1을 장비 0~M-1에 고정 ----------------
  for j in range(M):
      m = j
      model.Add(is_on[j][m] == 1)          # 반드시 해당 머신
      model.Add(start[j] == 0)             # 첫 작업, 시작 0
      # 다른 Job보다 앞섬: priority literal을 안 써도 NoOverlap이 보장
      # (Setup time은 뒤 Job이 알아서 고려)

  # ------------- NoOverlap + Setup time -----------------------
  for m in range(M):
      # NoOverlap 자체는 간섭 방지만 하고 setup time은 precedence-literal로 표현
      if intervals[m]:
          model.AddNoOverlap(intervals[m])

  # precedence literal + setup
  setup_used = {}
  priority_violation = []
  for m in range(M):
      for i in range(J):
          if (not matrixEquip[m][i]): continue
          for j in range(i+1, J):
              if not matrixEquip[m][j]: continue
              before_ij = model.NewBoolVar(f'before_{i}_{j}_{m}')
              # setup_flag = model.NewBoolVar(f'setup_used_{i}_{j}')
              diff = model.NewIntVar(0, setup_val, 'diff')
              # model.Add(matrixSeq[i][j] + matrixSeq[j][i] == 1)
              # Priority
              if i >= M: 
                  if priority[i] < priority[j]:
                      # priority_violation.append(matrixSeq[i][j])
                      priority_violation.append(before_ij)
                  else:
                      # priority_violation.append(matrixSeq[j][i])
                      priority_violation.append(before_ij.Not())

              # model.Add(setup_flag == matrixPcard[i][j]).OnlyEnforceIf(is_on[i][m], is_on[j][m])
              model.Add(diff == setup_val * matrixPcard[i][j]).OnlyEnforceIf(is_on[i][m], is_on[j][m])
              # model.Add(diff == setup_val).OnlyEnforceIf(setup_flag)
              setup_used[(i, j, m)] = diff

              # diff = setup_val * setup_flag
              model.Add(start[j] >= end[i] + diff).OnlyEnforceIf(
                  # [matrixSeq[i][j], is_on[i][m], is_on[j][m]])
                  [before_ij, is_on[i][m], is_on[j][m]])
              # j before i
              model.Add(start[i] >= end[j] + diff).OnlyEnforceIf(
                  # [matrixSeq[j][i], is_on[i][m], is_on[j][m]])
                  [before_ij.Not(), is_on[i][m], is_on[j][m]])

              

  ############ max C + Soft Penalty ############
  alpha = 10
  beta = 10
  maxC = model.NewIntVar(0, H, 'maxC')
  model.AddMaxEquality(maxC, end)
  # model.Minimize(maxC)
  model.Minimize(maxC + alpha * sum(priority_violation) + beta * sum(setup_used.values()))
  
  
  proto = model.Proto()
  print(f'Number of Variables: {len(proto.variables)}')
  print(f'Number of constraints: {len(proto.constraints)}')
  # ------------- Solver 설정 ----------------------------------
  solver = cp_model.CpSolver()
  # solver.parameters.max_time_in_seconds = 300
  solver.parameters.num_search_workers  = 8   # 병렬
  solver.parameters.log_search_progress = True
  solver.parameters.cp_model_probing_level = 2
  solver.parameters.cp_model_presolve = True
  
  status = solver.Solve(model)
  print(solver.StatusName(status))
  print(f'maxC: {solver.Value(maxC)}')
  print(f'prior: {sum(solver.Value(v) for v in priority_violation)}')
  print(f'setup: {sum(solver.Value(v) for v in setup_used.values())}')
  # print(f'setup: {sum(v for v in setup_used.values())}')
  # ------------- 결과 -----------------------------------------
  if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
      for j in range(J):
          m = next(m for m in range(M) if solver.Value(is_on[j][m]))
          s = solver.Value(start[j])/SCALE
          e = solver.Value(end[j])  /SCALE
          # print(f'Job {j:02d} | M{m:02d} | {pcard[j]} | start={s:.1f}, end={e:.1f} | p={p[j]/SCALE:.1f}')
      print('Avg Completion Time:', solver.Value(maxC)/SCALE)
  else:
      print('No solution found')
