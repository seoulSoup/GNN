  def pcardParser(tensorInput):
      if torch.equal(tensorInput, torch.tensor([1,0,0,0,0,0,1])): return 'FF44'
      elif torch.equal(tensorInput, torch.tensor([0,1,0,0,0,0,1])): return 'AV44'
      elif torch.equal(tensorInput, torch.tensor([0,0,0,1,0,1,0])): return 'JC30'
      else: return 'EMPT'

  # ----------------------- 입력 데이터 -----------------------
  M = 14                     # machine 개수
  tensorQueue = sample[:40]  # 최대 40개만 사용
  J = tensorQueue.size(0)

  SCALE      = 10                            # 0.1 단위
  dur_float  = tensorQueue[:, -1].numpy()    # 처리시간(float)
  p          = [int(max(v, 0.1) * SCALE) for v in dur_float]  # 0 이하면 최소 0.1

  priority   = tensorQueue[:, 9].numpy()
  pcard      = [pcardParser(t[2:9]) for t in tensorQueue]     # 'FF44' 등
  setup_val  = int(0.16 * SCALE)                              # setup time 정수

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
  for i in range(J):
      for j in range(M, J):
          if i < j:
              prec_ij = model.NewBoolVar(f'prec_{i}_before_{j}')
              prec_ji = model.NewBoolVar(f'prec_{j}_before_{i}')
              # model.Add(prec_ij + prec_ji == 1)
              model.Add(prec_ij + prec_ji <= 1)
              if i < M:
                  diff = setup_val if pcard[i] != pcard[j] else 0
                  # i before j
                  model.Add(start[j] >= end[i] + diff).OnlyEnforceIf(
                      [prec_ij, is_on[i][i], is_on[j][i]])
                      
              else:
                  # 두 Job이 같은 머신에 배치될 수 있는 경우만 literal 생성
                  common_m = [m for m in range(M) if matrixEquip[m][i] and matrixEquip[m][j]]
                  if not common_m:
                      continue
                  for m in common_m:
                      diff = setup_val if pcard[i] != pcard[j] else 0
                      # i before j
                      model.Add(start[j] >= end[i] + diff).OnlyEnforceIf(
                          [prec_ij, is_on[i][m], is_on[j][m]])
                      # j before i
                      model.Add(start[i] >= end[j] + diff).OnlyEnforceIf(
                          [prec_ji, is_on[i][m], is_on[j][m]])
					  # priority
                      if priority[i] < priority[j]:
                        model.Add(prec_ji == 1)
                      else:
                        model.Add(prec_ij == 1)
  # ------------- 우선순위 soft penalty ------------------------
  # penalties = []
  # for i in range(J):
  #     for j in range(i+1, J):
  #         if priority[i] < priority[j]:
  #             lit = model.NewBoolVar(f'prio_{i}_before_{j}')
  #             model.Add(prec_ij == lit)  # 이미 prec_ij 존재
  #             penalties.append(lit.Not())  # 위반 시 1

  # ------------- 목적 함수 ------------------------------------
  # avgC   = model.NewIntVar(0, H*5, 'avgC')
  # model.Add(avgC == sum(end))
  # alpha = 5   # priority penalty weight
  # model.Minimize(avgC + alpha * sum(penalties))
  # model.Minimize(avgC)
  maxC = model.NewIntVar(0, H, 'maxC')
  model.AddMaxEquality(maxC, end)
  model.Minimize(maxC)
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

  # ------------- 결과 -----------------------------------------
  if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
      for j in range(J):
          m = next(m for m in range(M) if solver.Value(is_on[j][m]))
          s = solver.Value(start[j])/SCALE
          e = solver.Value(end[j])  /SCALE
          print(f'Job {j:02d} | M{m:02d} | {pcard[j]} | start={s:.1f}, end={e:.1f}')
      print('Avg Completion Time:', solver.Value(maxC)/SCALE)
  else:
      print('No solution found')
