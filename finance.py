"""
Finance Calculators
====================
- Landed cost (import duties for India)
- NPV / IRR
- Depreciation (SLM, WDV, UoP)
- WACC
"""
import math


def calc_landed_cost(data):
    """Calculate total landed cost for an imported material."""
    cif_value = data.get('cif_value', 0)
    fob_value = data.get('fob_value', 0)
    freight = data.get('freight', 0)
    insurance_pct = data.get('insurance_pct', 0.5)  # % of CIF
    bcd_pct = data.get('bcd_pct', 10)  # Basic Customs Duty %
    sws_pct = data.get('sws_pct', 10)  # Social Welfare Surcharge % of BCD
    igst_pct = data.get('igst_pct', 18)
    cha_charges = data.get('cha_charges', 5000)
    port_handling = data.get('port_handling', 8000)
    local_transport = data.get('local_transport', 0)
    exchange_rate = data.get('exchange_rate', 84)  # INR per unit foreign currency
    foreign_value = data.get('foreign_value', 0)  # price in foreign currency
    gst_registered = data.get('gst_registered', True)

    # Calculate CIF if not provided directly
    if cif_value <= 0:
        if foreign_value > 0:
            fob_inr = foreign_value * exchange_rate
        else:
            fob_inr = fob_value

        if fob_inr <= 0:
            return {'error': 'Provide either CIF value, FOB value, or foreign value + exchange rate'}

        insurance = fob_inr * insurance_pct / 100
        cif_value = fob_inr + freight + insurance

    assessable_value = cif_value

    # Duties
    bcd = assessable_value * bcd_pct / 100
    sws = bcd * sws_pct / 100
    igst_base = assessable_value + bcd + sws
    igst = igst_base * igst_pct / 100
    total_duty = bcd + sws + igst

    # Total landed
    total_landed = cif_value + total_duty + cha_charges + port_handling + local_transport

    # Net cost (after IGST ITC recovery)
    itc_recovery = igst if gst_registered else 0
    net_landed = total_landed - itc_recovery

    return {
        'assessable_value': round(assessable_value, 2),
        'bcd': round(bcd, 2),
        'bcd_pct': bcd_pct,
        'sws': round(sws, 2),
        'igst_base': round(igst_base, 2),
        'igst': round(igst, 2),
        'igst_pct': igst_pct,
        'total_duty': round(total_duty, 2),
        'cha_charges': cha_charges,
        'port_handling': port_handling,
        'local_transport': local_transport,
        'total_landed': round(total_landed, 2),
        'itc_recovery': round(itc_recovery, 2),
        'net_landed': round(net_landed, 2),
        'effective_duty_pct': round((net_landed - cif_value) / max(cif_value, 1) * 100, 2),
    }


def calc_npv(data):
    """Calculate NPV, IRR, payback period."""
    cash_flows = data.get('cash_flows', [])  # [CF0, CF1, CF2, ...]
    wacc = data.get('wacc', 0.10)  # discount rate

    if not cash_flows:
        return {'error': 'No cash flows provided'}

    # NPV
    npv = sum(cf / (1 + wacc) ** t for t, cf in enumerate(cash_flows))

    # IRR (Newton-Raphson)
    irr = _calc_irr(cash_flows)

    # Payback (simple)
    cumulative = 0
    payback_simple = None
    for t, cf in enumerate(cash_flows):
        cumulative += cf
        if cumulative >= 0 and payback_simple is None:
            payback_simple = t
            break

    # Payback (discounted)
    cumulative_disc = 0
    payback_disc = None
    for t, cf in enumerate(cash_flows):
        cumulative_disc += cf / (1 + wacc) ** t
        if cumulative_disc >= 0 and payback_disc is None:
            payback_disc = t
            break

    # DSCR (if debt service provided)
    debt_service = data.get('annual_debt_service', 0)
    net_operating_income = data.get('net_operating_income', 0)
    dscr = round(net_operating_income / debt_service, 2) if debt_service > 0 else None

    return {
        'npv': round(npv, 2),
        'irr': round(irr * 100, 2) if irr is not None else None,
        'payback_simple': payback_simple,
        'payback_discounted': payback_disc,
        'wacc': round(wacc * 100, 2),
        'dscr': dscr,
        'decision': 'INVEST' if npv > 0 else 'REJECT',
    }


def _calc_irr(cash_flows, tol=1e-6, max_iter=200):
    """Newton-Raphson IRR."""
    if not cash_flows or len(cash_flows) < 2:
        return None
    r = 0.1  # initial guess
    for _ in range(max_iter):
        npv = sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cash_flows))
        if abs(dnpv) < 1e-12:
            break
        r_new = r - npv / dnpv
        if abs(r_new - r) < tol:
            return r_new
        r = r_new
        if abs(r) > 10:  # diverged
            return None
    return r


def calc_depreciation(data):
    """Calculate depreciation schedule."""
    purchase_price = data.get('purchase_price', 0)
    residual_value = data.get('residual_value', 0)
    useful_life = data.get('useful_life', 10)  # years
    method = data.get('method', 'SLM')  # SLM, WDV, UoP
    total_units = data.get('total_units', 0)  # for UoP
    annual_units = data.get('annual_units', [])  # for UoP

    depreciable = purchase_price - residual_value
    schedule = []

    if method == 'SLM':
        annual = depreciable / useful_life
        bv = purchase_price
        for yr in range(1, useful_life + 1):
            dep = annual
            bv -= dep
            schedule.append({
                'year': yr,
                'depreciation': round(dep, 2),
                'book_value': round(max(bv, residual_value), 2),
            })

    elif method == 'WDV':
        rate = data.get('wdv_rate', 0.20)
        bv = purchase_price
        for yr in range(1, useful_life + 1):
            dep = bv * rate
            if bv - dep < residual_value:
                dep = bv - residual_value
            bv -= dep
            schedule.append({
                'year': yr,
                'depreciation': round(dep, 2),
                'book_value': round(bv, 2),
            })

    elif method == 'UoP':
        if total_units <= 0:
            return {'error': 'Total units required for UoP method'}
        rate_per_unit = depreciable / total_units
        bv = purchase_price
        for yr, units in enumerate(annual_units, 1):
            dep = rate_per_unit * units
            if bv - dep < residual_value:
                dep = bv - residual_value
            bv -= dep
            schedule.append({
                'year': yr,
                'units': units,
                'depreciation': round(dep, 2),
                'book_value': round(bv, 2),
            })

    return {
        'method': method,
        'purchase_price': purchase_price,
        'residual_value': residual_value,
        'useful_life': useful_life,
        'depreciable_amount': round(depreciable, 2),
        'schedule': schedule,
    }


def calc_wacc(data):
    """Calculate Weighted Average Cost of Capital."""
    equity_pct = data.get('equity_pct', 50)  # % of total capital
    debt_pct = data.get('debt_pct', 50)
    cost_equity = data.get('cost_equity', 12)  # %
    cost_debt = data.get('cost_debt', 8)  # % (pre-tax)
    tax_rate = data.get('tax_rate', 25)  # %

    e = equity_pct / 100
    d = debt_pct / 100
    re = cost_equity / 100
    rd = cost_debt / 100
    t = tax_rate / 100

    wacc = e * re + d * rd * (1 - t)

    return {
        'wacc': round(wacc * 100, 2),
        'equity_component': round(e * re * 100, 2),
        'debt_component': round(d * rd * (1 - t) * 100, 2),
    }
