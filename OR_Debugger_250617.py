def debug_constraints(obj, constraints, variables, solver='HIGHS'):
    print(f"전체 제약조건 수: {len(constraints)}")
    infeasible_indices = []

    for i, c in enumerate(constraints):
        reduced_constraints = constraints[:i] + constraints[i+1:]  # i번째 조건 제외
        prob = cp.Problem(obj, reduced_constraints)

        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status == 'infeasible':
                infeasible_indices.append(i)
                print(f"[{i:>3}] 제외 → 여전히 infeasible")
            elif prob.status in ['unbounded', 'unbounded_inaccurate']:
                print(f"[{i:>3}] 제외 → 문제 unbounded")
            else:
                print(f"[{i:>3}] 제외 → 해결 가능 (status: {prob.status}) ✅")
        except Exception as e:
            print(f"[{i:>3}] 오류 발생: {e}")
    
    if not infeasible_indices:
        print("\n⚠️ 모든 제약을 제거해도 여전히 infeasible하지 않거나, 특정 제약만의 문제는 아닐 수 있음.")
    else:
        print(f"\n❗ 유력한 문제 제약 조건 인덱스: {infeasible_indices}")