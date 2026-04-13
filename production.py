"""
Production Scheduler MILP
==========================
Given a set of production orders (products × quantities), production lines,
and setup/changeover costs between products:

Minimize: total setup cost + overtime cost + makespan penalty
Subject to: capacity, sequence, changeover time

Outputs: Gantt chart data, sequence per line per period, utilization
"""
import pulp
import time
import math


def solve_production(data):
    t0 = time.time()
    products = data.get('products', [])
    lines = data.get('lines', [])
    params = data.get('params', {})

    T = params.get('periods', 26)
    ot_cost = params.get('overtime_cost_per_hr', 50)
    hrs_per_shift = params.get('hours_per_shift', 8)
    makespan_weight = params.get('makespan_weight', 0.1)

    n_prod = len(products)
    n_lines = len(lines) if lines else 1

    if not n_prod:
        return {'error': 'No products'}

    # If no lines defined, create a default shared line
    if not lines:
        lines = [{'id': 'line1', 'name': 'Line 1', 'capacity': 50,
                   'type': 'shared', 'products': list(range(n_prod))}]
        n_lines = 1

    prob = pulp.LpProblem("Production_Scheduler", pulp.LpMinimize)

    # Decision variables
    # x[k,l,t] = units of product k on line l in period t
    x = {}
    # y[k,l,t] = binary: product k assigned to line l in period t
    y = {}
    # ot[l,t] = overtime hours on line l in period t
    ot = {}
    # completion[k] = last period product k is produced
    completion = {}

    for k in range(n_prod):
        completion[k] = pulp.LpVariable(f'comp_{k}', 0, T, cat='Integer')
        for l in range(n_lines):
            for t in range(T):
                cap = lines[l].get('capacity', 50)
                x[k, l, t] = pulp.LpVariable(f'x_{k}_{l}_{t}', 0, cap, cat='Integer')
                y[k, l, t] = pulp.LpVariable(f'y_{k}_{l}_{t}', cat='Binary')
            # Changeover: w[k1,k2,l,t] = 1 if switch from k1 to k2 on line l at period t
    # Changeover variables (simplified: count active product switches per line per period)
    switch = {}
    for l in range(n_lines):
        for t in range(1, T):
            switch[l, t] = pulp.LpVariable(f'sw_{l}_{t}', 0, cat='Binary')

    for l in range(n_lines):
        for t in range(T):
            ot[l, t] = pulp.LpVariable(f'ot_{l}_{t}', 0, hrs_per_shift)

    # Objective: minimize setup + overtime + makespan
    obj = []
    for k in range(n_prod):
        setup_cost = products[k].get('setup_cost', 50)
        for l in range(n_lines):
            for t in range(T):
                obj.append(setup_cost * y[k, l, t])

    # Changeover cost
    changeover_cost = params.get('changeover_cost', 100)
    for l in range(n_lines):
        for t in range(1, T):
            obj.append(changeover_cost * switch[l, t])

    # Overtime cost
    for l in range(n_lines):
        for t in range(T):
            obj.append(ot_cost * ot[l, t])

    # Makespan penalty (encourage finishing early)
    for k in range(n_prod):
        obj.append(makespan_weight * completion[k])

    prob += pulp.lpSum(obj)

    # Constraints
    for k in range(n_prod):
        req = products[k].get('required_qty', 100)
        fy = products[k].get('yield_pct', 0.95)

        # C1: Total production meets requirement
        prob += pulp.lpSum(
            x[k, l, t] for l in range(n_lines) for t in range(T)
        ) * fy >= req, f"Demand_{k}"

        # C2: Can only produce on assigned lines
        for l in range(n_lines):
            eligible = lines[l].get('products', list(range(n_prod)))
            if k not in eligible and isinstance(eligible[0] if eligible else 0, int):
                for t in range(T):
                    prob += x[k, l, t] == 0, f"Inelig_{k}_{l}_{t}"

        # C3: Linking x and y
        for l in range(n_lines):
            cap = lines[l].get('capacity', 50)
            for t in range(T):
                prob += x[k, l, t] <= cap * y[k, l, t], f"Link_{k}_{l}_{t}"
                prob += x[k, l, t] >= y[k, l, t], f"MinProd_{k}_{l}_{t}"

        # C4: Completion tracking
        for l in range(n_lines):
            for t in range(T):
                prob += completion[k] >= (t + 1) * y[k, l, t], f"Comp_{k}_{l}_{t}"

    # C5: Line capacity per period (sum across products)
    for l in range(n_lines):
        cap = lines[l].get('capacity', 50)
        shifts = lines[l].get('shifts_per_day', 1)
        total_cap = cap * shifts
        for t in range(T):
            # Regular capacity + overtime extension
            ot_cap_extra = cap * 0.5  # OT can add 50% more
            prob += pulp.lpSum(
                x[k, l, t] for k in range(n_prod)
            ) <= total_cap + ot_cap_extra * (ot[l, t] / hrs_per_shift), f"LineCap_{l}_{t}"

    # C6: Shared lines — max 1 active product per period (optional for sequential mode)
    for l in range(n_lines):
        if lines[l].get('type') == 'shared':
            for t in range(T):
                prob += pulp.lpSum(
                    y[k, l, t] for k in range(n_prod)
                ) <= 2, f"Shared_{l}_{t}"  # allow 2 for changeover periods

    # C7: Changeover detection
    for l in range(n_lines):
        for t in range(1, T):
            for k in range(n_prod):
                # If product k was NOT on line l at t-1 but IS at t → switch
                prob += switch[l, t] >= y[k, l, t] - y[k, l, t - 1], f"SwDet_{k}_{l}_{t}"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60, gapRel=0.05)
    status = prob.solve(solver)
    solve_time = time.time() - t0

    if status != pulp.constants.LpStatusOptimal:
        return {'status': pulp.LpStatus[status], 'error': f'Solver: {pulp.LpStatus[status]}',
                'solve_time': round(solve_time, 2)}

    total_cost = pulp.value(prob.objective)

    # Extract Gantt data
    gantt = []
    product_results = []
    for k in range(n_prod):
        prod_total = 0
        prod_periods = []
        for t in range(T):
            for l in range(n_lines):
                qty = int(pulp.value(x[k, l, t]) or 0)
                if qty > 0:
                    gantt.append({
                        'product': products[k].get('name', f'P{k}'),
                        'line': lines[l].get('name', f'L{l}'),
                        'line_idx': l,
                        'period': t,
                        'quantity': qty,
                        'product_idx': k,
                    })
                    prod_total += qty
                    prod_periods.append(t)

        comp = int(pulp.value(completion[k]) or 0)
        product_results.append({
            'name': products[k].get('name', f'P{k}'),
            'required': products[k].get('required_qty', 100),
            'produced': prod_total,
            'completion_period': comp,
            'active_periods': len(prod_periods),
            'utilization': round(len(prod_periods) / max(T, 1) * 100, 1),
        })

    # Line utilization
    line_results = []
    for l in range(n_lines):
        active = sum(1 for t in range(T) if any(
            int(pulp.value(x[k, l, t]) or 0) > 0 for k in range(n_prod)))
        total_produced = sum(
            int(pulp.value(x[k, l, t]) or 0)
            for k in range(n_prod) for t in range(T))
        cap = lines[l].get('capacity', 50)
        ot_hrs = sum(pulp.value(ot[l, t]) or 0 for t in range(T))
        line_results.append({
            'name': lines[l].get('name', f'L{l}'),
            'active_periods': active,
            'utilization': round(active / max(T, 1) * 100, 1),
            'total_produced': total_produced,
            'overtime_hours': round(ot_hrs, 1),
            'changeovers': sum(int(pulp.value(switch.get((l, t), 0)) or 0) for t in range(1, T)),
        })

    return {
        'status': 'Optimal',
        'total_cost': round(total_cost, 2),
        'solve_time': round(solve_time, 2),
        'products': product_results,
        'lines': line_results,
        'gantt': gantt,
        'periods': T,
    }
