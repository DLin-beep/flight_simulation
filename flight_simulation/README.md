# Flight Decision Dashboard: Shift-Robust Risk-Constrained Routing

A decision engine for airline-style routing under uncertainty. Given an origin/destination pair, the system evaluates top-*k* candidate itineraries and selects routes that trade off **expected cost**, **tail risk (CVaR\_0.95)**, and **CO\_2**, with optional **chance constraints** on on-time performance.

## Core ideas

- **Risk-constrained optimization**: select routes under constraints like \(\mathbb{P}(\text{delay}\le 15) \ge \tau\) and CO\_2 budgets.
- **Tail-risk scoring**: uses **CVaR\_0.95** (average of the worst 5% outcomes) as an objective/constraint.
- **Shift-robust uncertainty quantification**: supports **conformal prediction** and **weighted conformal calibration** to preserve interval coverage under time-split distribution shift when real delay data is provided.
- **Pareto frontier**: produces non-dominated routes over (expected cost, CVaR\_0.95, mean CO\_2).

## Quickstart

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

### CLI (offline demo)

```bash
python main.py route --orig JFK --dest LHR --objective cvar --k 8 --samples 4000 --offline
python experiments/resilience.py --orig JFK --dest LHR --trials 200 --closure_prob 0.01 --seed 0
```

## Optional: upload real delay data

The dashboard can ingest a CSV of flight delays to fit a conditional delay model and evaluate calibration under time-based shift.

Expected columns (case-insensitive):
- `ORIGIN` (IATA code)
- `DEST` (IATA code)
- `ARR_DELAY` (minutes; negative values are clipped to 0)
- `FL_DATE` (recommended; enables time-split shift evaluation)

If `FL_DATE` is missing, the app falls back to a non-temporal split.

## Project structure

- `app.py` — Streamlit decision dashboard
- `src/` — routing, uncertainty, calibration, and multi-objective selection
- `experiments/` — scripted evaluations (shift robustness, resilience)
- `data/` — offline sample airports/routes

## License

MIT (see `LICENSE`).
