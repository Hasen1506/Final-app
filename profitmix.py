"""
Product Mix / Profit Maximizer LP
==================================
Given products with different contribution margins and shared capacity constraints,
determine the optimal product mix that maximizes total profit.

Also computes shadow prices (dual values) for binding constraints — showing
the value of adding one more unit of each scarce resource.
"""
import pulp
import time


def solve_profitmix(data):
    t0 = time.time()
    products = data.get('products', [])
    constraints = data.get('constraints', {})

    n = len(products)
    if not n:
        return {'error': 'No products'}

    prob = pulp.LpProblem("Profit_Maximizer", pulp.LpMaximize)

    # Decision variables: quantity of each product to produce
    q = {}
    for k in range(n):
        max_demand = products[k].get('max_demand', 99999)
        q[k] = pulp.LpVariable(f'q_{k}', 0, max_demand, cat='Continuous')

    # Objective: maximize total contribution margin
    # Contribution = selling price - variable cost - material cost
    obj = []
    for k in range(n):
        p = products[k]
        sell = p.get('sell_price', 100)
        var_cost = p.get('variable_cost', 0)
        mat_cost = sum(
            part.get('cost', 0) * part.get('qty_per', 1)
            for part in p.get('parts', [])
        )
        margin = sell - var_cost - mat_cost
        products[k]['_margin'] = margin
        obj.append(margin * q[k])

    prob += pulp.lpSum(obj), "Total_Profit"

    # Constraints
    constraint_names = {}

    # C1: Shared capacity (hours or units)
    shared_cap = constraints.get('shared_capacity', 0)
    if shared_cap > 0:
        cycle_times = [p.get('cycle_time', 1) for p in products]
        c = prob.addConstraint(
            pulp.lpSum(cycle_times[k] * q[k] for k in range(n)) <= shared_cap,
            "Shared_Capacity"
        )
        constraint_names['Shared Capacity'] = "Shared_Capacity"

    # C2: Per-line capacity
    lines = constraints.get('lines', [])
    for li, line in enumerate(lines):
        cap = line.get('capacity', 999999)
        line_products = line.get('products', list(range(n)))
        c_name = f"Line_{line.get('name', li)}"
        prob += pulp.lpSum(
            q[k] for k in line_products if k < n
        ) <= cap, c_name
        constraint_names[f"Line: {line.get('name', f'L{li}')}"] = c_name

    # C3: Budget constraint
    budget = constraints.get('budget', 0)
    if budget > 0:
        prob += pulp.lpSum(
            (products[k].get('variable_cost', 0) +
             sum(pt.get('cost', 0) * pt.get('qty_per', 1) for pt in products[k].get('parts', []))) * q[k]
            for k in range(n)
        ) <= budget, "Budget"
        constraint_names['Budget'] = "Budget"

    # C4: Material availability
    materials = constraints.get('materials', {})
    for mat_name, avail in materials.items():
        # Find which products use this material
        mat_users = []
        for k in range(n):
            for part in products[k].get('parts', []):
                if part.get('name', '') == mat_name:
                    mat_users.append((k, part.get('qty_per', 1)))
        if mat_users:
            c_name = f"Mat_{mat_name}"
            prob += pulp.lpSum(
                qty_per * q[k] for k, qty_per in mat_users
            ) <= avail, c_name
            constraint_names[f"Material: {mat_name}"] = c_name

    # C5: Minimum production (contractual commitments)
    for k in range(n):
        min_qty = products[k].get('min_quantity', 0)
        if min_qty > 0:
            prob += q[k] >= min_qty, f"MinQty_{k}"

    # C6: Warehouse/storage
    wh_max = constraints.get('warehouse', 0)
    if wh_max > 0:
        prob += pulp.lpSum(q[k] for k in range(n)) <= wh_max, "Warehouse"
        constraint_names['Warehouse'] = "Warehouse"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    status = prob.solve(solver)
    solve_time = time.time() - t0

    if status != pulp.constants.LpStatusOptimal:
        return {'status': pulp.LpStatus[status], 'error': f'Solver: {pulp.LpStatus[status]}',
                'solve_time': round(solve_time, 2)}

    total_profit = pulp.value(prob.objective)

    # Extract results
    product_results = []
    total_revenue = 0
    total_cost = 0
    for k in range(n):
        qty = pulp.value(q[k]) or 0
        p = products[k]
        margin = p.get('_margin', 0)
        revenue = qty * p.get('sell_price', 0)
        cost = qty * (p.get('variable_cost', 0) + sum(
            pt.get('cost', 0) * pt.get('qty_per', 1) for pt in p.get('parts', [])))
        profit = qty * margin
        total_revenue += revenue
        total_cost += cost

        product_results.append({
            'name': p.get('name', f'P{k}'),
            'quantity': round(qty, 1),
            'margin_per_unit': round(margin, 2),
            'revenue': round(revenue, 2),
            'cost': round(cost, 2),
            'profit': round(profit, 2),
            'pct_of_total': round(profit / max(total_profit, 1) * 100, 1),
            'max_demand': p.get('max_demand', 99999),
            'demand_filled': round(qty / max(p.get('max_demand', qty), 1) * 100, 1) if p.get('max_demand') else 100,
        })

    # Shadow prices (dual values) for constraints
    shadow_prices = []
    for display_name, c_name in constraint_names.items():
        c = prob.constraints.get(c_name)
        if c is not None:
            dual = c.pi if hasattr(c, 'pi') and c.pi is not None else 0
            slack = c.slack if hasattr(c, 'slack') else None
            binding = slack is not None and abs(slack) < 0.01
            shadow_prices.append({
                'constraint': display_name,
                'shadow_price': round(dual, 2),
                'slack': round(slack, 2) if slack is not None else None,
                'binding': binding,
                'interpretation': (
                    f"Adding 1 more unit increases profit by ${abs(round(dual, 2))}"
                    if binding and dual != 0
                    else f"Not binding — {round(slack, 1)} units unused" if slack else "—"
                ),
            })

    return {
        'status': 'Optimal',
        'total_profit': round(total_profit, 2),
        'total_revenue': round(total_revenue, 2),
        'total_cost': round(total_cost, 2),
        'margin_pct': round(total_profit / max(total_revenue, 1) * 100, 1),
        'products': product_results,
        'shadow_prices': shadow_prices,
        'solve_time': round(solve_time, 2),
    }
