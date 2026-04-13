"""
Transport Optimizer
====================
Two sub-problems:

1. MODE SELECTION: For each shipment, choose optimal transport mode
   (road/rail/sea/air) considering cost, time, urgency, and volume.

2. ALLOCATION: Given supply at origins and demand at destinations,
   minimize total shipping cost (transportation problem LP).

Demand sensing integration: if a spike is detected and slow modes
won't deliver in time, recommends faster mode with cost premium.
"""
import pulp
import time
import math


def solve_transport(data):
    t0 = time.time()
    shipments = data.get('shipments', [])
    params = data.get('params', {})

    if not shipments:
        return {'error': 'No shipments provided'}

    # ─── MODE SELECTION ───
    modes = {
        'road': {'cost_per_kg': 2.5, 'cost_per_cbm': 800, 'transit_days': 5,
                  'min_weight': 0, 'max_weight': 25000, 'reliability': 0.90},
        'rail': {'cost_per_kg': 1.5, 'cost_per_cbm': 500, 'transit_days': 8,
                  'min_weight': 1000, 'max_weight': 60000, 'reliability': 0.85},
        'sea_lcl': {'cost_per_kg': 0.8, 'cost_per_cbm': 300, 'transit_days': 25,
                     'min_weight': 0, 'max_weight': 14000, 'reliability': 0.88},
        'sea_fcl': {'cost_per_kg': 0.4, 'cost_per_cbm': 150, 'transit_days': 22,
                     'min_weight': 5000, 'max_weight': 26000, 'reliability': 0.90},
        'air': {'cost_per_kg': 15, 'cost_per_cbm': 4000, 'transit_days': 3,
                 'min_weight': 0, 'max_weight': 5000, 'reliability': 0.97},
    }

    # Override with user-provided rates
    user_modes = params.get('modes', {})
    for mode_name, overrides in user_modes.items():
        if mode_name in modes:
            modes[mode_name].update(overrides)

    results = []
    total_cost = 0
    total_weight = 0
    mode_summary = {m: {'count': 0, 'cost': 0, 'weight': 0} for m in modes}

    for ship in shipments:
        weight = ship.get('weight_kg', 100)
        volume = ship.get('volume_cbm', 1)
        value = ship.get('value', 10000)
        deadline_days = ship.get('deadline_days', 30)
        urgency = ship.get('urgency', 'normal')  # normal, high, critical
        origin = ship.get('origin', 'Factory')
        destination = ship.get('destination', 'Customer')
        item_name = ship.get('name', 'Shipment')

        # Demand sensing flag
        demand_spike = ship.get('demand_spike', False)
        spike_qty = ship.get('spike_qty', 0)

        # Evaluate each mode
        mode_options = []
        for mode_name, mode in modes.items():
            # Check feasibility
            if weight < mode['min_weight'] or weight > mode['max_weight']:
                continue
            if mode['transit_days'] > deadline_days:
                continue  # can't meet deadline

            # Calculate cost (use chargeable weight: max of actual vs volumetric)
            vol_weight = volume * 167 if mode_name == 'air' else volume * 1000 / 6
            chargeable = max(weight, vol_weight)
            cost = chargeable * mode['cost_per_kg']

            # Urgency penalty for slow modes
            buffer_days = deadline_days - mode['transit_days']
            risk_score = max(0, 1 - buffer_days / max(deadline_days, 1))
            if urgency == 'critical' and buffer_days < 3:
                cost *= 0.8  # discount fast modes for critical shipments (prefer them)
            elif urgency == 'high' and buffer_days < 5:
                cost *= 0.9

            # Stockout cost if late
            daily_revenue = value / 30
            stockout_risk = (1 - mode['reliability']) * daily_revenue * mode['transit_days']
            total_mode_cost = cost + stockout_risk

            mode_options.append({
                'mode': mode_name,
                'cost': round(cost, 2),
                'total_cost': round(total_mode_cost, 2),
                'transit_days': mode['transit_days'],
                'buffer_days': buffer_days,
                'reliability': mode['reliability'],
                'stockout_risk': round(stockout_risk, 2),
                'chargeable_weight': round(chargeable, 1),
            })

        # Sort by total cost
        mode_options.sort(key=lambda x: x['total_cost'])

        recommended = mode_options[0] if mode_options else None
        cheapest = mode_options[0] if mode_options else None
        fastest = min(mode_options, key=lambda x: x['transit_days']) if mode_options else None

        # Demand spike override: if spike detected and recommended is slow, suggest air
        spike_recommendation = None
        if demand_spike and recommended and recommended['transit_days'] > 7:
            air_option = next((m for m in mode_options if m['mode'] == 'air'), None)
            if air_option:
                premium = air_option['cost'] - recommended['cost']
                spike_recommendation = {
                    'message': f"Demand spike detected (+{spike_qty}u). Sea/road won't arrive in time.",
                    'recommended_mode': 'air',
                    'cost_premium': round(premium, 2),
                    'time_saved': recommended['transit_days'] - air_option['transit_days'],
                    'justified': premium < value * 0.1,  # premium < 10% of shipment value
                }

        ship_cost = recommended['total_cost'] if recommended else 0
        total_cost += ship_cost
        total_weight += weight

        if recommended:
            mode_summary[recommended['mode']]['count'] += 1
            mode_summary[recommended['mode']]['cost'] += ship_cost
            mode_summary[recommended['mode']]['weight'] += weight

        results.append({
            'name': item_name,
            'origin': origin,
            'destination': destination,
            'weight_kg': weight,
            'volume_cbm': volume,
            'value': value,
            'deadline_days': deadline_days,
            'urgency': urgency,
            'recommended': recommended,
            'cheapest': cheapest,
            'fastest': fastest,
            'all_options': mode_options,
            'spike_alert': spike_recommendation,
        })

    # ─── ALLOCATION (if origins/destinations provided) ───
    allocation_result = None
    origins = data.get('origins', [])
    destinations = data.get('destinations', [])
    cost_matrix = data.get('cost_matrix', [])

    if origins and destinations and cost_matrix:
        alloc = _solve_allocation(origins, destinations, cost_matrix)
        allocation_result = alloc

    solve_time = time.time() - t0

    return {
        'status': 'Optimal',
        'total_cost': round(total_cost, 2),
        'total_weight': round(total_weight, 1),
        'shipments': results,
        'mode_summary': {k: v for k, v in mode_summary.items() if v['count'] > 0},
        'allocation': allocation_result,
        'solve_time': round(solve_time, 2),
    }


def _solve_allocation(origins, destinations, cost_matrix):
    """Solve the transportation problem: minimize total shipping cost."""
    n_orig = len(origins)
    n_dest = len(destinations)

    prob = pulp.LpProblem("Transport_Allocation", pulp.LpMinimize)

    # x[i,j] = quantity shipped from origin i to destination j
    x = {}
    for i in range(n_orig):
        for j in range(n_dest):
            x[i, j] = pulp.LpVariable(f'x_{i}_{j}', 0)

    # Objective: minimize total cost
    prob += pulp.lpSum(
        cost_matrix[i][j] * x[i, j]
        for i in range(n_orig) for j in range(n_dest)
    )

    # Supply constraints
    for i in range(n_orig):
        prob += pulp.lpSum(
            x[i, j] for j in range(n_dest)
        ) <= origins[i].get('supply', 0), f"Supply_{i}"

    # Demand constraints
    for j in range(n_dest):
        prob += pulp.lpSum(
            x[i, j] for i in range(n_orig)
        ) >= destinations[j].get('demand', 0), f"Demand_{j}"

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=10)
    status = prob.solve(solver)

    if status != pulp.constants.LpStatusOptimal:
        return {'error': f'Allocation solver: {pulp.LpStatus[status]}'}

    # Extract allocation
    allocation = []
    for i in range(n_orig):
        for j in range(n_dest):
            qty = pulp.value(x[i, j]) or 0
            if qty > 0.5:
                allocation.append({
                    'from': origins[i].get('name', f'O{i}'),
                    'to': destinations[j].get('name', f'D{j}'),
                    'quantity': round(qty, 1),
                    'unit_cost': cost_matrix[i][j],
                    'total_cost': round(qty * cost_matrix[i][j], 2),
                })

    return {
        'total_cost': round(pulp.value(prob.objective), 2),
        'allocation': allocation,
    }
