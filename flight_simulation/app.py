from __future__ import annotations



import hashlib

import json

from dataclasses import asdict

from typing import Dict, List, Tuple



import numpy as np

import pandas as pd

import streamlit as st





try:

    import plotly.graph_objects as go

except Exception:

    go = None



from src.calibration import interval_coverage, pinball_loss

from src.cost_analyzer import FlightCostAnalyzer

from src.data_loader import FlightDataLoader

from src.multiobjective import RouteMetrics, feasible_set, pareto_frontier, pick_by_weights

from src.route_finder import RouteFinder

from src.srcr import ShiftRobustConformalDelayModel







DARK_CSS = """
<style>
:root { --bg: #0b1020; --card: #111a2f; --muted: #9aa7c2; --text: #e9eefc; --accent: #7aa2ff; }
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
html, body, [class*="css"]  { background-color: var(--bg) !important; color: var(--text) !important; }
h1,h2,h3,h4 { color: var(--text) !important; }
.small-muted { color: var(--muted); font-size: 0.9rem; }
.pill { display:inline-block; padding:6px 10px; margin-right:8px; border-radius:999px; background:#121a33; border:1px solid #1f2a4a; color: var(--muted); font-size: 0.85rem;}
.card-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin-top: 10px; }
.card { background: var(--card); border: 1px solid #1f2a4a; border-radius: 16px; padding: 14px 14px; }
.card-title { color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }
.card-value { font-size: 1.35rem; font-weight: 700; color: var(--text); }
.card-sub { color: var(--muted); font-size: 0.85rem; margin-top: 6px;}
hr { border-color: #1f2a4a !important; }
</style>
"""





def stable_seed(*parts: str) -> int:

    h = hashlib.md5("||".join(parts).encode("utf-8")).hexdigest()

    return int(h[:8], 16)





def _fmt_money(x: float) -> str:

    try:

        return f"${float(x):,.0f}"

    except Exception:

        return "—"





def _fmt_int(x: float) -> str:

    try:

        return f"{float(x):,.0f}"

    except Exception:

        return "—"





def _scalar(x) -> float:

    if x is None:

        return float("nan")

    if isinstance(x, (list, tuple, np.ndarray)):

        if len(x) == 0:

            return float("nan")

        return float(x[0])

    return float(x)





def card(title: str, value: str, subtitle: str = ""):

    st.markdown(

        f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="card-value">{value}</div>
          <div class="card-sub">{subtitle}</div>
        </div>
        """,

        unsafe_allow_html=True,

    )





def build_route_map(airports: pd.DataFrame, path: List[str]):

    """Plot route on world map using Scattergeo. Returns None if Plotly unavailable."""

    if go is None or not path or len(path) < 2:

        return None

    df = airports.copy()

    df["iata"] = df["iata"].astype(str).str.strip().str.upper()

    if "lat" not in df.columns or "lon" not in df.columns:

        return None

    coords = df.set_index("iata")[["lat", "lon"]].to_dict("index")



    lats, lons, labels = [], [], []

    for code in path:

        c = coords.get(str(code).strip().upper())

        if not c:

            continue

        lats.append(float(c["lat"]))

        lons.append(float(c["lon"]))

        labels.append(str(code).strip().upper())



    if len(lats) < 2:

        return None



    fig = go.Figure()

    fig.add_trace(

        go.Scattergeo(

            lon=lons,

            lat=lats,

            mode="lines+markers+text",

            text=labels,

            textposition="top center",

            line=dict(width=2),

            marker=dict(size=8),

            hoverinfo="text",

        )

    )

    fig.update_layout(

        margin=dict(l=0, r=0, t=0, b=0),

        geo=dict(

            scope="world",

            projection_type="natural earth",

            showland=True,

            landcolor="rgb(30, 30, 35)",

            showocean=True,

            oceancolor="rgb(15, 18, 25)",

            showcountries=True,

            countrycolor="rgb(70, 70, 80)",

            lataxis_showgrid=False,

            lonaxis_showgrid=False,

        ),

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        showlegend=False,

    )

    return fig





@st.cache_data(show_spinner=False)

def load_assets(offline: bool):

    loader = FlightDataLoader(allow_download=not offline)

    net_raw = loader.load_network()





    if isinstance(net_raw, dict):

        graph = net_raw.get("graph")

        edge_distance_km = net_raw.get("edge_distance_km") or {}

    else:

        graph = getattr(net_raw, "graph", None)

        edge_distance_km = getattr(net_raw, "edge_distance_km", {}) or {}



    if graph is None:

        raise RuntimeError("Network graph failed to load. Ensure data/sample_routes.csv exists for offline mode.")



    airports = loader.load_airport_data().copy()

    airports["iata"] = airports["iata"].astype(str).str.strip().str.upper()

    airports = airports[airports["iata"].str.len() == 3].drop_duplicates(subset=["iata"])



    def label_row(r) -> str:

        iata = str(r.get("iata", "")).strip().upper()

        name = str(r.get("name", "")).strip()

        city = str(r.get("city", "")).strip()

        country = str(r.get("country", "")).strip()

        bits = [b for b in [name, city, country] if b and b.lower() != "nan"]

        pretty = " — ".join(bits) if bits else iata

        return f"{iata} | {pretty}"



    airports["label"] = airports.apply(label_row, axis=1)

    label_to_iata = dict(zip(airports["label"], airports["iata"]))

    iata_to_label = dict(zip(airports["iata"], airports["label"]))



    rf = RouteFinder(graph, edge_distance_km=edge_distance_km)

    return airports, label_to_iata, iata_to_label, rf, edge_distance_km





def metrics_to_df(ms: List[RouteMetrics]) -> pd.DataFrame:

    rows = []

    for m in ms:

        d = asdict(m)

        d["path"] = " → ".join(d["path"])

        rows.append(d)

    df = pd.DataFrame(rows)

    if df.empty:

        return df

    df["stops"] = df["path"].apply(lambda s: max(0, s.count("→") - 1))

    return df





def preset_to_weights(preset: str) -> Dict[str, float]:

    if preset == "Lowest cost":

        return {"w_cost": 0.70, "w_risk": 0.20, "w_co2": 0.10}

    if preset == "Most reliable":

        return {"w_cost": 0.25, "w_risk": 0.65, "w_co2": 0.10}

    if preset == "Lowest emissions":

        return {"w_cost": 0.25, "w_risk": 0.15, "w_co2": 0.60}

    return {"w_cost": 0.50, "w_risk": 0.30, "w_co2": 0.20}





def build_decision_report(best: RouteMetrics, preset: str, constraints: Dict[str, float]) -> Dict:

    return {

        "pitch": (

            "A calibrated probabilistic model for flight time/cost/emissions uncertainty, "

            "used to compute Pareto-optimal routes under tail-risk constraints (CVaR) and CO₂ budgets, "

            "with validated coverage and decision reports."

        ),

        "decision": {

            "preset": preset,

            "constraints": constraints,

            "recommended_path": list(best.path),

            "metrics": asdict(best),

        },

        "explain": [

            "We evaluate multiple feasible routes (k-shortest candidates).",

            "For each route we simulate cost/time/CO₂ under a heavy-tailed uncertainty model (or real delay data if uploaded).",

            "We compute tail risk using CVaR95 and filter to Pareto-optimal tradeoffs.",

            "We select a recommendation using an interpretable objective preset and optional budgets.",

        ],

    }





def run_calibration_tables(seed: int, uploaded: pd.DataFrame | None, focus_airport: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """Return (coverage summary, pinball loss detail) for Baseline/Vanilla/Weighted conformal.
    If no upload, returns a self-contained synthetic demo.
    """

    if uploaded is not None and len(uploaded) > 0:

        model = ShiftRobustConformalDelayModel(alpha=0.1, min_group=30, seed=seed).fit(uploaded)

        return model.coverage_table(focus_airport=focus_airport), model.detail_table(focus_airport=focus_airport)





    rng = np.random.default_rng(seed)

    y_cal = rng.lognormal(mean=2.0, sigma=0.6, size=400) * (rng.random(400) < 0.35)

    y_te = rng.lognormal(mean=2.1, sigma=0.75, size=400) * (rng.random(400) < 0.40)





    def baseline_interval(samples, alpha=0.1):

        lo = float(np.quantile(samples, alpha / 2.0))

        hi = float(np.quantile(samples, 1.0 - alpha / 2.0))

        return lo, hi



    cal_lo, cal_hi = baseline_interval(y_cal)

    te_lo, te_hi = baseline_interval(y_te)





    scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)

    q = float(np.quantile(scores, 0.9))

    te_lo_c, te_hi_c = te_lo - q, te_hi + q



    summary = pd.DataFrame(

        [

            {"method": "Baseline (uncalibrated)", "target": 0.90, "achieved": float(interval_coverage(y_te, te_lo, te_hi))},

            {"method": "Vanilla Conformal", "target": 0.90, "achieved": float(interval_coverage(y_te, te_lo_c, te_hi_c))},

        ]

    )

    metrics = pd.DataFrame(

        [

            {"method": "Baseline (uncalibrated)", "median_pinball": float(pinball_loss(y_te, np.full_like(y_te, (te_lo + te_hi) / 2.0), q=0.5))},

            {"method": "Vanilla Conformal", "median_pinball": float(pinball_loss(y_te, np.full_like(y_te, (te_lo_c + te_hi_c) / 2.0), q=0.5))},

        ]

    )

    metrics["dataset"] = f"Synthetic demo (airport={focus_airport})"

    return summary, metrics







st.set_page_config(page_title="Flight Simulation", layout="wide")

st.markdown(DARK_CSS, unsafe_allow_html=True)



st.markdown(

    """
    <div class="pill">Calibrated uncertainty</div>
    <div class="pill">CVaR tail-risk</div>
    <div class="pill">CO₂ budgets</div>
    <div class="pill">Shift-robust (weighted conformal)</div>
    """,

    unsafe_allow_html=True,

)



st.markdown("## Flight Decision Dashboard")

st.markdown('<div class="small-muted">Nontechnical-friendly routing recommendations with calibrated risk & emissions tradeoffs.</div>', unsafe_allow_html=True)



airports, label_to_iata, iata_to_label, rf, edge_distance_km = load_assets(offline=True)





with st.sidebar:

    st.header("Decision settings")

    preset = st.selectbox("Goal", ["Balanced", "Lowest cost", "Most reliable", "Lowest emissions"], index=0)

    st.caption("Tip: choose a goal, optionally add budgets, then click **Run**.")



    st.divider()

    st.subheader("Optional: real delay data")

    uploaded_file = st.file_uploader("Upload flight delay CSV (optional)", type=["csv"])

    st.caption("Expected columns: ORIGIN, DEST, ARR_DELAY, and optionally FL_DATE.")



    st.divider()

    st.subheader("Budgets (optional)")

    max_co2 = st.number_input("Max mean CO₂ (kg)", min_value=0.0, value=0.0, step=1000.0)

    max_cvar = st.number_input("Max CVaR95 cost (USD)", min_value=0.0, value=0.0, step=1000.0)

    min_otp = st.slider("Min on-time probability", 0.0, 1.0, 0.0, step=0.01)



    st.divider()

    k = st.slider("Candidate routes (k)", 3, 25, 12)

    samples = st.slider("Monte Carlo samples", 500, 12000, 4000, step=500)

    stop_penalty_km = st.slider("Connection penalty (km)", 0.0, 1500.0, 250.0, step=50.0)

    seed_ui = st.number_input("Seed", min_value=0, value=0, step=1)



    run = st.button("Run", type="primary", use_container_width=True)





uploaded_df = None

if uploaded_file is not None:

    try:

        uploaded_df = pd.read_csv(uploaded_file)

    except Exception:

        st.sidebar.error("Could not read CSV. Please upload a valid comma-separated file.")

        uploaded_df = None



labels = airports["label"].tolist()



default_orig = next((l for l in labels if l.startswith("JFK |")), labels[0] if labels else "")

default_dest = next((l for l in labels if l.startswith("LHR |")), labels[1] if len(labels) > 1 else default_orig)



c1, c2 = st.columns(2)

with c1:

    orig_label = st.selectbox("Origin", labels, index=labels.index(default_orig) if default_orig in labels else 0)

with c2:

    dest_label = st.selectbox("Destination", labels, index=labels.index(default_dest) if default_dest in labels else min(1, len(labels) - 1))



orig = label_to_iata.get(orig_label, orig_label.split("|")[0].strip().upper())

dest = label_to_iata.get(dest_label, dest_label.split("|")[0].strip().upper())



if not run:

    st.info("Choose a goal on the left, then click **Run**.")

    st.stop()





delay_sampler = None

if uploaded_df is not None and len(uploaded_df) > 0:

    try:

        srcr = ShiftRobustConformalDelayModel(alpha=0.1, min_group=30, seed=int(seed_ui)).fit(uploaded_df)

        delay_sampler = srcr.make_delay_sampler()

    except Exception:

        st.warning("Could not fit SRCR delay model from uploaded data. Falling back to synthetic delay model.")



analyzer = FlightCostAnalyzer(edge_distance_km=edge_distance_km, stop_penalty_usd=5000.0, seed=int(seed_ui))



with st.spinner("Simulating candidate routes..."):

    cands = rf.k_shortest_paths(orig, dest, k=int(k), per_leg_penalty_km=float(stop_penalty_km))

    metrics: List[RouteMetrics] = []

    for pr in cands:

        s = analyzer.summarize_path(pr.path, samples=int(samples), delay_sampler=delay_sampler)

        metrics.append(

            RouteMetrics(

                path=tuple(pr.path),

                distance_km=float(pr.distance_km),

                mean_cost_usd=float(s.mean_usd),

                p90_cost_usd=float(s.p90_usd),

                cvar95_cost_usd=float(s.cvar_usd),

                mean_time_min=float(s.mean_min),

                p90_time_min=float(s.p90_min),

                mean_co2_kg=float(s.mean_co2_kg),

                p90_co2_kg=float(s.p90_co2_kg),

                on_time_prob=float(s.on_time_prob),

            )

        )



df_all = metrics_to_df(metrics)

if df_all.empty:

    st.error(f"No route found from {orig} to {dest}.")

    st.stop()



constraints = {"max_mean_co2_kg": float(max_co2), "max_cvar95_cost_usd": float(max_cvar), "min_on_time_prob": float(min_otp)}

feasible = feasible_set(metrics, max_mean_co2_kg=float(max_co2), max_cvar95_cost_usd=float(max_cvar), min_on_time_prob=float(min_otp))

frontier = pareto_frontier(feasible if feasible else metrics)



df_feasible = metrics_to_df(feasible if feasible else metrics)

df_frontier = metrics_to_df(frontier)



weights = preset_to_weights(preset)

best = pick_by_weights(frontier, **weights)





st.markdown("### Recommendation")

path_str = " → ".join(best.path)

st.markdown(f"**{path_str}**")

st.markdown(f'<div class="small-muted">{max(0, len(best.path)-2)} connections | {_fmt_int(best.distance_km)} km</div>', unsafe_allow_html=True)



st.markdown('<div class="card-grid">', unsafe_allow_html=True)

colA, colB, colC, colD = st.columns(4)

with colA:

    card("Expected cost", _fmt_money(best.mean_cost_usd), f"p90: {_fmt_money(best.p90_cost_usd)}")

with colB:

    card("Tail-risk (CVaR95)", _fmt_money(best.cvar95_cost_usd), "Average of worst 5% outcomes")

with colC:

    card("CO₂ (mean)", f"{_fmt_int(best.mean_co2_kg)} kg", f"p90: {_fmt_int(best.p90_co2_kg)} kg")

with colD:

    card("On-time probability", f"{best.on_time_prob*100:.0f}%", "P(delay ≤ 15 min)")

st.markdown("</div>", unsafe_allow_html=True)





if feasible == [] and (max_co2 > 0 or max_cvar > 0 or min_otp > 0):

    st.warning("No routes satisfy the selected budgets. Showing the Pareto frontier without budgets.")



st.markdown("#### Route map")

fig = build_route_map(airports, list(best.path))

if fig is None:

    st.info("Map view requires Plotly. Install it with: `python -m pip install plotly`.")

else:

    st.plotly_chart(fig, use_container_width=True)





st.markdown("### Top alternatives")

df_show = df_frontier.sort_values(["cvar95_cost_usd", "mean_cost_usd"]).head(8).copy()

st.dataframe(

    df_show[["path", "stops", "distance_km", "mean_cost_usd", "cvar95_cost_usd", "mean_co2_kg", "on_time_prob"]],

    use_container_width=True,

    hide_index=True,

)



if go is not None and not df_frontier.empty:

    st.markdown("### Tradeoffs (Pareto frontier)")

    f = go.Figure()

    f.add_trace(

        go.Scatter(

            x=df_frontier["mean_co2_kg"],

            y=df_frontier["cvar95_cost_usd"],

            mode="markers",

            marker=dict(size=10),

            text=df_frontier["path"],

            hovertemplate="CO₂=%{x:.0f} kg<br>CVaR95=%{y:.0f} USD<br>%{text}<extra></extra>",

        )

    )

    f.update_layout(

        xaxis_title="Mean CO₂ (kg)",

        yaxis_title="CVaR95 Cost (USD)",

        margin=dict(l=0, r=0, t=10, b=0),

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

    )

    st.plotly_chart(f, use_container_width=True)





st.markdown("### Research: shift-robust calibration (Weighted Conformal)")

st.markdown(

    '<div class="small-muted">'

    "We evaluate three methods on a time-split holdout (distribution shift): "

    "<b>Baseline</b> (uncalibrated), <b>Vanilla Conformal</b>, and <b>Weighted Conformal</b> "

    "(importance-weighted conformal scores using a learned shift classifier)."

    "</div>",

    unsafe_allow_html=True,

)



summary, detail = run_calibration_tables(seed=stable_seed("calib", str(seed_ui)), uploaded=uploaded_df, focus_airport=orig)

cA, cB = st.columns(2)

with cA:

    st.markdown("**Shift evaluation (coverage)**")

    st.dataframe(summary, use_container_width=True, hide_index=True)

with cB:

    st.markdown("**Quality metrics / examples**")

    st.dataframe(detail, use_container_width=True, hide_index=True)





st.markdown("### Decision report")

report = build_decision_report(best, preset=preset, constraints=constraints)

st.download_button("Download report (JSON)", data=json.dumps(report, indent=2).encode("utf-8"), file_name="decision_report.json")

st.download_button("Download Pareto frontier (CSV)", data=df_frontier.to_csv(index=False).encode("utf-8"), file_name="pareto_frontier.csv")

st.download_button("Download feasible set (CSV)", data=df_feasible.to_csv(index=False).encode("utf-8"), file_name="feasible_routes.csv")



st.markdown("---")

st.markdown(

    '<div class="small-muted">Tip: For real novelty, upload a delay CSV with FL_DATE so calibration is evaluated on a future (shifted) period. The routing engine will then use the fitted SRCR delay sampler per leg.</div>',

    unsafe_allow_html=True,

)

