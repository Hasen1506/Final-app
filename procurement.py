"""
Procurement & Production MILP Solver
=====================================
Minimizes total supply chain cost:
  setup + FG holding + production + expiry + shortage + RM purchase + RM holding + RM ordering

Decision variables:
  p[k,t]  = units of product k produced in period t
  r[i,t]  = units of raw material i ordered in period t
  y[k,t]  = binary: whether product k is produced in period t
  o[i,t]  = binary: whether raw material i is ordered in period t
"""
import pulp
import math
import time
import numpy as np


def solve_procurement(data):
    """Main entry point. data = dict from API request."""
    t0 = time.time()
    products = data.get('products', [])
    params = data.get('params', {})
    cap_mode = data.get('capacity_mode', 'parallel')  # shared or parallel

    T = params.get('periods', 52)  # weekly periods
    carry_rate = params.get('carry_rate', 0.24)
    wh_max = params.get('wh_max', 5000)
    fixed_daily = params.get('fixed_daily', 0)
    bo_on = params.get('backorder_on', False)
    salvage = params.get('salvage_rate', 0.80)
    service_level = params.get('service_level', 0.95)
    budget = params.get('budget', None)  # optional budget constraint

    n_products = len(products)
    if not n_products:
        return {'error': 'No products provided'}

    z_map = {0.85: 1.036, 0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_map.get(service_level, 1.645)

    # ── Build problem ──
    prob = pulp.LpProblem("Procurement_Optimizer", pulp.LpMinimize)

    # ── Decision variables ──
    p = {}   # production
    inv = {}  # FG inventory
    short = {}  # shortages
    y = {}   # production binary
    r = {}   # RM orders
    o = {}   # RM order binary
    rm_inv = {}  # RM inventory

    all_parts = []
    part_map = {}  # (product_idx, part_idx) -> global part index

    for k, prod in enumerate(products):
        demand = prod.get('demand', [0] * T)
        while len(demand) < T:
            demand.append(demand[-1] if demand else 0)
        demand = demand[:T]

        cap = prod.get('capacity', 50)
        setup_cost = prod.get('setup_cost', 50)
        var_cost = prod.get('variable_cost', 0)
        shelf = prod.get('shelf_life', T)
        sell_price = prod.get('sell_price', 10)
        fy = prod.get('yield_pct', 0.95)
        short_penalty = sell_price * 1.5  # lost margin + goodwill

        # Safety stock
        demand_arr = np.array(demand, dtype=float)
        avg_d = max(demand_arr.mean(), 0.1)
        std_d = max(demand_arr.std(), 0.1)
        ss = max(1, round(z * std_d))

        for t in range(T):
            p[k, t] = pulp.LpVariable(f'p_{k}_{t}', 0, cap, cat='Integer')
            inv[k, t] = pulp.LpVariable(f'inv_{k}_{t}', 0)
            short[k, t] = pulp.LpVariable(f'short_{k}_{t}', 0)
            y[k, t] = pulp.LpVariable(f'y_{k}_{t}', cat='Binary')

        # BOM parts
        parts = prod.get('parts', [])
        for i, part in enumerate(parts):
            gidx = len(all_parts)
            part_map[(k, i)] = gidx
            all_parts.append({
                'name': part.get('name', f'Part_{k}_{i}'),
                'cost': part.get('cost', 1.0),
                'qty_per': part.get('qty_per', 1.0),
                'lt': part.get('lead_time', 1),
                'moq': part.get('moq', 1),
                'max_order': part.get('max_order', 9999),
                'hold_pct': part.get('hold_pct', carry_rate * 100),
                'rm_cap': part.get('rm_capacity', 9999),
                'ord_cost': part.get('ordering_cost', 50),
                'rm_shelf': part.get('rm_shelf', T),
                'product_k': k,
                'part_i': i,
                'scrap': part.get('scrap_factor', 0),
            })

            for t in range(T):
                r[gidx, t] = pulp.LpVariable(f'r_{gidx}_{t}', 0, cat='Integer')
                o[gidx, t] = pulp.LpVariable(f'o_{gidx}_{t}', cat='Binary')
                rm_inv[gidx, t] = pulp.LpVariable(f'rminv_{gidx}_{t}', 0)

    # ── Objective function ──
    obj = []

    for k, prod in enumerate(products):
        demand = prod.get('demand', [0] * T)[:T]
        setup_cost = prod.get('setup_cost', 50)
        var_cost = prod.get('variable_cost', 0)
        sell_price = prod.get('sell_price', 10)
        shelf = prod.get('shelf_life', T)
        fy = prod.get('yield_pct', 0.95)
        unit_cost = sum(
            pt.get('cost', 1) * pt.get('qty_per', 1)
            for pt in prod.get('parts', [])
        )
        fg_hold = unit_cost * carry_rate / 52  # weekly holding cost per unit
        short_penalty = sell_price * 1.5

        for t in range(T):
            # Setup cost
            obj.append(setup_cost * y[k, t])
            # Variable production cost
            obj.append(var_cost * p[k, t])
            # FG holding
            obj.append(fg_hold * inv[k, t])
            # Shortage penalty
            obj.append(short_penalty * short[k, t])

    # RM costs
    for gidx, part in enumerate(all_parts):
        for t in range(T):
            # Purchase cost
            obj.append(part['cost'] * r[gidx, t])
            # Ordering admin cost
            obj.append(part['ord_cost'] * o[gidx, t])
            # RM holding
            rm_hold = part['cost'] * (part['hold_pct'] / 100) / 52
            obj.append(rm_hold * rm_inv[gidx, t])

    # Fixed overhead
    obj.append(fixed_daily * T)

    prob += pulp.lpSum(obj), "Total_Cost"

    # ── Constraints ──

    for k, prod in enumerate(products):
        demand = prod.get('demand', [0] * T)[:T]
        cap = prod.get('capacity', 50)
        shelf = prod.get('shelf_life', T)
        fy = prod.get('yield_pct', 0.95)
        init_inv = prod.get('init_inventory', 0)

        demand_arr = np.array(demand, dtype=float)
        ss = max(1, round(z * max(demand_arr.std(), 0.1)))

        for t in range(T):
            d = demand[t] if t < len(demand) else demand[-1]

            # C1: Inventory balance
            prev_inv = inv[k, t - 1] if t > 0 else init_inv
            good_prod = p[k, t]  # simplified: yield applied at BOM consumption
            prob += inv[k, t] == prev_inv + good_prod - d + short[k, t], \
                f"InvBal_{k}_{t}"

            # C2: Capacity
            prob += p[k, t] <= cap * y[k, t], f"Cap_{k}_{t}"

            # C3: Min production (if producing, produce at least 1)
            prob += p[k, t] >= y[k, t], f"MinProd_{k}_{t}"

            # C4: Warehouse limit (shared across products)
            # Added below as aggregate

            # C5: No backorder if disabled
            if not bo_on:
                prob += short[k, t] == 0, f"NoBO_{k}_{t}"

    # Aggregate warehouse constraint
    for t in range(T):
        prob += pulp.lpSum(
            inv[k, t] for k in range(n_products)
        ) <= wh_max, f"WH_{t}"

    # Shared capacity constraint
    if cap_mode == 'shared':
        shared_cap = params.get('shared_capacity', 100)
        for t in range(T):
            prob += pulp.lpSum(
                p[k, t] for k in range(n_products)
            ) <= shared_cap, f"SharedCap_{t}"

    # RM constraints
    for gidx, part in enumerate(all_parts):
        k = part['product_k']
        i = part['part_i']
        prod = products[k]
        demand = prod.get('demand', [0] * T)[:T]
        lt = part['lt']
        moq = part['moq']
        max_ord = part['max_order']
        rm_cap = part['rm_cap']
        qty_per = part['qty_per']
        scrap = part['scrap']
        fy = prod.get('yield_pct', 0.95)
        effective_qty = qty_per * (1 + scrap) / max(fy, 0.01)
        # Default init RM: enough for lead_time periods of avg demand
        avg_demand_per_t = sum(demand[:T]) / T if T > 0 else 10
        default_init_rm = max(0, round(avg_demand_per_t * effective_qty * (lt + 1)))
        init_rm = part.get('init_inventory', default_init_rm)

        for t in range(T):
            # RM arrives lt periods after ordering
            arrive_t = t - lt
            arrived = r[gidx, arrive_t] if arrive_t >= 0 else 0

            # RM consumption = production * effective qty per unit
            consumed = p[k, t] * effective_qty

            prev_rm = rm_inv[gidx, t - 1] if t > 0 else init_rm
            prob += rm_inv[gidx, t] == prev_rm + arrived - consumed, \
                f"RMBal_{gidx}_{t}"

            # RM non-negative (redundant with var bounds but explicit)
            prob += rm_inv[gidx, t] >= 0, f"RMNonNeg_{gidx}_{t}"

            # MOQ: if ordering, order at least MOQ
            prob += r[gidx, t] >= moq * o[gidx, t], f"MOQ_{gidx}_{t}"
            prob += r[gidx, t] <= max_ord * o[gidx, t], f"MaxOrd_{gidx}_{t}"

            # RM warehouse capacity
            prob += rm_inv[gidx, t] <= rm_cap, f"RMCap_{gidx}_{t}"

    # Budget constraint (optional)
    if budget and budget > 0:
        prob += pulp.lpSum(
            part['cost'] * r[gidx, t]
            for gidx, part in enumerate(all_parts)
            for t in range(T)
        ) <= budget, "Budget"

    # ── Solve ──
    solver = pulp.PULP_CBC_CMD(
        msg=0,
        timeLimit=90,
        gapRel=0.02
    )
    status = prob.solve(solver)

    solve_time = time.time() - t0

    if status != pulp.constants.LpStatusOptimal:
        return {
            'status': pulp.LpStatus[status],
            'error': f'Solver returned: {pulp.LpStatus[status]}',
            'solve_time': round(solve_time, 2)
        }

    # ── Extract results ──
    total_cost = pulp.value(prob.objective)

    product_results = []
    for k, prod in enumerate(products):
        demand = prod.get('demand', [0] * T)[:T]
        prod_schedule = [int(pulp.value(p[k, t]) or 0) for t in range(T)]
        inv_levels = [round(pulp.value(inv[k, t]) or 0, 1) for t in range(T)]
        shortages = [round(pulp.value(short[k, t]) or 0, 1) for t in range(T)]
        setups = [int(pulp.value(y[k, t]) or 0) for t in range(T)]

        total_prod = sum(prod_schedule)
        total_demand = sum(demand[:T])
        total_short = sum(shortages)
        fill_rate = round((1 - total_short / max(total_demand, 1)) * 100, 1)

        product_results.append({
            'name': prod.get('name', f'Product_{k}'),
            'production': prod_schedule,
            'inventory': inv_levels,
            'shortages': shortages,
            'setups': setups,
            'total_produced': total_prod,
            'total_demand': total_demand,
            'total_shortage': round(total_short),
            'fill_rate': fill_rate,
            'num_batches': sum(setups),
        })

    material_results = []
    for gidx, part in enumerate(all_parts):
        orders = [int(pulp.value(r[gidx, t]) or 0) for t in range(T)]
        rm_levels = [round(pulp.value(rm_inv[gidx, t]) or 0, 1) for t in range(T)]
        order_flags = [int(pulp.value(o[gidx, t]) or 0) for t in range(T)]

        # Build PO list
        po_list = []
        for t in range(T):
            if orders[t] > 0:
                po_list.append({
                    'period': t,
                    'arrive_period': t + part['lt'],
                    'quantity': orders[t],
                    'cost': round(orders[t] * part['cost'], 2),
                })

        material_results.append({
            'name': part['name'],
            'product': products[part['product_k']].get('name', ''),
            'orders': orders,
            'inventory': rm_levels,
            'purchase_orders': po_list,
            'total_ordered': sum(orders),
            'total_cost': round(sum(orders) * part['cost'], 2),
            'num_orders': sum(order_flags),
        })

    # Cost breakdown
    cost_breakdown = {
        'total': round(total_cost, 2),
        'material_purchase': round(sum(
            sum(int(pulp.value(r[g, t]) or 0) * all_parts[g]['cost']
                for t in range(T))
            for g in range(len(all_parts))
        ), 2),
        'ordering_admin': round(sum(
            sum(int(pulp.value(o[g, t]) or 0) * all_parts[g]['ord_cost']
                for t in range(T))
            for g in range(len(all_parts))
        ), 2),
        'production_setup': round(sum(
            sum(int(pulp.value(y[k, t]) or 0) * products[k].get('setup_cost', 50)
                for t in range(T))
            for k in range(n_products)
        ), 2),
        'production_variable': round(sum(
            sum(int(pulp.value(p[k, t]) or 0) * products[k].get('variable_cost', 0)
                for t in range(T))
            for k in range(n_products)
        ), 2),
        'fixed_overhead': round(fixed_daily * T, 2),
    }

    return {
        'status': 'Optimal',
        'total_cost': round(total_cost, 2),
        'cost_breakdown': cost_breakdown,
        'products': product_results,
        'materials': material_results,
        'solve_time': round(solve_time, 2),
        'periods': T,
        'solver': 'CBC',
    }
