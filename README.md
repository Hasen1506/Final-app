# Enterprise Simulator v2.0

Production-grade supply chain optimization platform with MILP solver, Monte Carlo simulation, and financial analytics.

## Quick Start (Local)
```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploy to Render
1. Push to GitHub
2. Connect repo on [render.com](https://render.com)
3. It auto-detects `render.yaml` — deploys as Web Service
4. Starter plan ($7/mo) is sufficient

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Server status |
| `/api/solve/procurement` | POST | MILP procurement optimizer (PuLP/CBC) |
| `/api/solve/montecarlo` | POST | 500-run stochastic simulation |
| `/api/solve/sensitivity` | POST | Parameter sweep analysis |
| `/api/solve/researcher` | POST | 150-experiment auto-optimizer |
| `/api/calc/landed-cost` | POST | Indian import duty calculator |
| `/api/calc/npv` | POST | NPV/IRR/payback calculator |
| `/api/calc/depreciation` | POST | SLM/WDV/UoP depreciation |
| `/api/calc/wacc` | POST | Weighted average cost of capital |

## Architecture
```
app.py              → Flask server (serves SPA + API)
static/index.html   → React SPA (17 forecast models, 7-tab UI)
solvers/
  procurement.py    → MILP: minimize procurement + inventory cost
  montecarlo.py     → Stochastic simulation (VaR/CVaR)
  finance.py        → NPV, landed cost, depreciation, WACC
```

## Phase Roadmap
- **Phase 1** ✅ Foundation: Flask + React + MILP + MC + Forecast Engine (17 models)
- **Phase 2** ✅ Supply Network (landed cost calc, Incoterms) + Finance (NPV/IRR, WACC, depreciation 3 methods, CCC) + Analysis (MC VaR/CVaR, sensitivity, auto-researcher, control tower, TCO waterfall)
- **Phase 3** ✅ Production Scheduler MILP (Gantt chart) + Profit Maximizer LP (shadow prices) + Transport Optimizer (mode selection + demand spike alerts)
- **Phase 4** 🔨 What-If Bot (Claude API), EVM tracker, scenario comparison
- **Phase 5** 🔨 Learning Lab (18+ sections) + Network visualization + PDF export
