"""
Enterprise Simulator v2.0 — Flask Backend
==========================================
Serves the React SPA + provides solver API endpoints.

Deploy to Render:
  git push → auto-deploys
  
Local:
  pip install -r requirements.txt
  python app.py
  → http://localhost:5000
"""
import os
import json
import time
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from solvers.procurement import solve_procurement
from solvers.production import solve_production
from solvers.profitmix import solve_profitmix
from solvers.transport import solve_transport
from solvers.montecarlo import run_montecarlo
from solvers.finance import calc_landed_cost, calc_npv, calc_depreciation, calc_wacc

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)


# ─── Static Frontend ───
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ─── Health Check ───
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'version': '2.0.0',
        'solver': 'PuLP CBC',
        'timestamp': time.time(),
    })


# ─── Procurement MILP Solver ───
@app.route('/api/solve/procurement', methods=['POST'])
def api_solve_procurement():
    try:
        data = request.json
        result = solve_procurement(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


# ─── Monte Carlo Simulation ───
@app.route('/api/solve/montecarlo', methods=['POST'])
def api_solve_montecarlo():
    try:
        data = request.json
        n_runs = data.get('n_runs', 500)
        result = run_montecarlo(data, n_runs=n_runs)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ─── Sensitivity Analysis ───
@app.route('/api/solve/sensitivity', methods=['POST'])
def api_solve_sensitivity():
    try:
        data = request.json
        base_data = data.get('base_data', {})
        param_ranges = data.get('param_ranges', {})
        results = []

        # For each parameter, sweep values and run MC
        for param_name, values in param_ranges.items():
            for val in values:
                modified = json.loads(json.dumps(base_data))
                # Apply parameter change
                if '.' in param_name:
                    parts = param_name.split('.')
                    obj = modified
                    for p in parts[:-1]:
                        obj = obj.get(p, {})
                    obj[parts[-1]] = val
                else:
                    modified['params'][param_name] = val

                mc_result = run_montecarlo(modified, n_runs=100)
                results.append({
                    'param': param_name,
                    'value': val,
                    'avg_cost': mc_result['avg_cost'],
                    'var95': mc_result['var95'],
                    'fill': mc_result['avg_fill'],
                })

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ─── Auto-Researcher (150 experiments) ───
@app.route('/api/solve/researcher', methods=['POST'])
def api_solve_researcher():
    try:
        data = request.json
        import numpy as np
        rng = np.random.default_rng()
        base_data = data.get('base_data', {})
        mode = data.get('mode', 'fixed')  # fixed or upgrade
        n_experiments = min(data.get('n_experiments', 150), 200)

        # Parameter ranges for exploration
        sl_values = [0.85, 0.90, 0.95, 0.99]
        setup_values = [20, 50, 100, 200]
        hold_delay = [0, 1, 3, 7]

        if mode == 'upgrade':
            cap_values = [20, 35, 50, 80]
            yield_values = [0.85, 0.90, 0.95, 1.0]
            shelf_values = [4, 8, 12, 26, 52]

        experiments = []
        for exp in range(n_experiments):
            modified = json.loads(json.dumps(base_data))
            config = {}

            # Randomize parameters
            sl = float(rng.choice(sl_values))
            modified['params']['service_level'] = sl
            config['service_level'] = sl

            setup = int(rng.choice(setup_values))
            for prod in modified.get('products', []):
                prod['setup_cost'] = setup
            config['setup_cost'] = setup

            if mode == 'upgrade':
                cap = int(rng.choice(cap_values))
                for prod in modified.get('products', []):
                    prod['capacity'] = cap
                config['capacity'] = cap

                fy = float(rng.choice(yield_values))
                for prod in modified.get('products', []):
                    prod['yield_pct'] = fy
                config['yield_pct'] = fy

                sh = int(rng.choice(shelf_values))
                for prod in modified.get('products', []):
                    prod['shelf_life'] = sh
                config['shelf_life'] = sh

            # Run MC (fast: 50 runs)
            mc = run_montecarlo(modified, n_runs=50)
            experiments.append({
                'config': config,
                'avg_cost': mc['avg_cost'],
                'var95': mc['var95'],
                'fill': mc['avg_fill'],
                'fragility': mc['fragility'],
            })

        # Sort by avg_cost
        experiments.sort(key=lambda x: x['avg_cost'])

        # Get baseline
        baseline_mc = run_montecarlo(base_data, n_runs=50)
        baseline_cost = baseline_mc['avg_cost']

        for exp in experiments:
            exp['savings_pct'] = round(
                (baseline_cost - exp['avg_cost']) / max(baseline_cost, 1) * 100, 1
            )

        return jsonify({
            'baseline_cost': baseline_cost,
            'experiments': experiments[:20],  # top 20
            'total_run': n_experiments,
            'mode': mode,
        })
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ─── Finance: Landed Cost ───
@app.route('/api/calc/landed-cost', methods=['POST'])
def api_landed_cost():
    try:
        return jsonify(calc_landed_cost(request.json))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Finance: NPV ───
@app.route('/api/calc/npv', methods=['POST'])
def api_npv():
    try:
        return jsonify(calc_npv(request.json))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Finance: Depreciation ───
@app.route('/api/calc/depreciation', methods=['POST'])
def api_depreciation():
    try:
        return jsonify(calc_depreciation(request.json))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Finance: WACC ───
@app.route('/api/calc/wacc', methods=['POST'])
def api_wacc():
    try:
        return jsonify(calc_wacc(request.json))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Production Scheduler MILP ───
@app.route('/api/solve/production', methods=['POST'])
def api_solve_production():
    try:
        return jsonify(solve_production(request.json))
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ─── Profit Maximizer LP ───
@app.route('/api/solve/profitmix', methods=['POST'])
def api_solve_profitmix():
    try:
        return jsonify(solve_profitmix(request.json))
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ─── Transport Optimizer ───
@app.route('/api/solve/transport', methods=['POST'])
def api_solve_transport():
    try:
        return jsonify(solve_transport(request.json))
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
