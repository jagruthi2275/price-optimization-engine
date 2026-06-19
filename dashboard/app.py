"""
Streamlit Dashboard for the Price Optimization Engine.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

st.set_page_config(
    page_title="Price Optimization Engine",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f1117; }
.kpi-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #63b3ed; }
.kpi-label { font-size: 0.8rem; color: #a0aec0; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.rec-box {
    background: linear-gradient(135deg, #1a365d 0%, #2a4365 100%);
    border-left: 4px solid #63b3ed;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 6px 0;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,17,23,0.85)",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
)


@st.cache_resource(show_spinner=False)
def _load_model_and_metrics():
    from src.model_training import load_best_model, run_training_pipeline, load_metrics
    model_path = os.path.join(BASE_DIR, "models", "best_model.joblib")
    if not os.path.exists(model_path):
        run_training_pipeline()
    model = load_best_model()
    metrics = load_metrics()
    return model, metrics


@st.cache_data(show_spinner=False, ttl=600)
def _load_eda():
    from src.data_preprocessing import load_raw_data, handle_missing_values, feature_engineering, run_eda
    df = load_raw_data()
    df = handle_missing_values(df)
    df = feature_engineering(df)
    return run_eda(df), df


def predict(price, comp_price, promo, stock, season, holiday, weekday, model):
    from src.data_preprocessing import preprocess_single
    X = preprocess_single(price, comp_price, promo, stock, season, holiday, weekday)
    return max(float(model.predict(X)[0]), 0)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 Price Optimizer")
    st.markdown("---")
    st.markdown("### Product Parameters")
    price = st.number_input("Selling Price ($)", 1.0, 5000.0, 100.0, 1.0)
    comp_price = st.number_input("Competitor Price ($)", 1.0, 5000.0, 95.0, 1.0)
    stock = st.number_input("Inventory Level (units)", 0, 50000, 500, 50)
    promo = st.selectbox("Promotion Active", [0, 1], format_func=lambda x: "Yes ✓" if x else "No")
    st.markdown("### Context")
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], index=1)
    holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: "Yes ✓" if x else "No")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = st.select_slider("Day of Week", options=list(range(7)), value=2,
                                format_func=lambda x: day_names[x])
    st.markdown("### Optimization Settings")
    n_sim = st.slider("Simulation Points", 20, 150, 60, 10)
    st.markdown("---")
    st.caption("Adjust inputs and navigate tabs to explore results.")


# ── Load Resources ─────────────────────────────────────────────────────────────
with st.spinner("Loading model (first run trains automatically)…"):
    model, metrics_summary = _load_model_and_metrics()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("💰 Price Optimization Engine")
st.markdown("*ML-powered demand forecasting · Revenue optimization · Price elasticity modeling*")
st.divider()

# ── KPI Row ────────────────────────────────────────────────────────────────────
demand_now = predict(price, comp_price, promo, stock, season, holiday, weekday, model)
revenue_now = price * demand_now
best_model_name = metrics_summary.get("best_model_name", "—") if metrics_summary else "—"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Current Price", f"${price:,.2f}")
k2.metric("Predicted Demand", f"{demand_now:,.0f} units")
k3.metric("Expected Revenue", f"${revenue_now:,.0f}")
k4.metric("vs Competitor", f"${price - comp_price:+.2f}", delta_color="inverse")
k5.metric("Best Model", best_model_name)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_forecast, tab_optimize, tab_elastic, tab_metrics, tab_eda, tab_shap, tab_retrain = st.tabs([
    "📈 Demand Forecast",
    "💡 Price Optimization",
    "🔄 Elasticity Analysis",
    "🤖 Model Metrics",
    "📊 Data Exploration",
    "🧠 Explainability",
    "⚙️ Retrain",
])


# ── TAB 1: Demand Forecast ─────────────────────────────────────────────────────
with tab_forecast:
    st.markdown("### Demand Sensitivity Curve")

    prices_sweep = np.linspace(price * 0.4, price * 1.6, 70)
    demands_sweep = [predict(p, comp_price, promo, stock, season, holiday, weekday, model) for p in prices_sweep]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices_sweep, y=demands_sweep, mode="lines",
        line=dict(color="#63b3ed", width=3),
        fill="tozeroy", fillcolor="rgba(99,179,237,0.10)",
        name="Predicted Demand",
    ))
    fig.add_vline(x=price, line_dash="dash", line_color="#f6ad55", line_width=2,
                  annotation_text=f"  Current ${price:.0f}", annotation_font_color="#f6ad55")
    fig.update_layout(**THEME, height=380, xaxis_title="Price ($)", yaxis_title="Predicted Demand (units)",
                      title="Demand vs. Price (±60% range)")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### Season Effect")
        s_vals = {s: predict(price, comp_price, promo, stock, s, holiday, weekday, model)
                  for s in ["Spring", "Summer", "Fall", "Winter"]}
        fig2 = go.Figure(go.Bar(
            x=list(s_vals.keys()), y=list(s_vals.values()),
            marker_color=["#68d391", "#f6ad55", "#fc8181", "#76e4f7"],
            text=[f"{v:,.0f}" for v in s_vals.values()], textposition="outside",
        ))
        fig2.update_layout(**THEME, height=280, showlegend=False, yaxis_title="Demand")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("#### Promotion Impact")
        d_off = predict(price, comp_price, 0, stock, season, holiday, weekday, model)
        d_on = predict(price, comp_price, 1, stock, season, holiday, weekday, model)
        fig3 = go.Figure(go.Bar(
            x=["No Promo", "With Promo"], y=[d_off, d_on],
            marker_color=["#718096", "#68d391"],
            text=[f"{d_off:,.0f}", f"{d_on:,.0f}"], textposition="outside",
        ))
        fig3.update_layout(**THEME, height=280, showlegend=False, yaxis_title="Demand")
        st.plotly_chart(fig3, use_container_width=True)
        uplift_pct = (d_on - d_off) / max(d_off, 1) * 100
        st.caption(f"Promotion uplift: **+{uplift_pct:.1f}%**")

    with col_c:
        st.markdown("#### Weekday Effect")
        day_demands = [predict(price, comp_price, promo, stock, season, holiday, d, model) for d in range(7)]
        fig4 = go.Figure(go.Bar(
            x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=day_demands,
            marker_color=["#63b3ed"] * 5 + ["#f6ad55", "#f6ad55"],
            text=[f"{v:,.0f}" for v in day_demands], textposition="outside",
        ))
        fig4.update_layout(**THEME, height=280, showlegend=False, yaxis_title="Demand")
        st.plotly_chart(fig4, use_container_width=True)


# ── TAB 2: Price Optimization ──────────────────────────────────────────────────
with tab_optimize:
    st.markdown("### Revenue Optimization Engine")

    with st.spinner("Running price simulation…"):
        from src.revenue_optimization import run_optimization
        opt = run_optimization(price, comp_price, promo, stock, model,
                               season=season, holiday=holiday, weekday=weekday, n_simulations=n_sim)

    opt_price = opt["recommended_price"]
    opt_revenue = opt["expected_revenue"]
    opt_demand = opt["predicted_demand"]
    uplift = opt["revenue_uplift_pct"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recommended Price", f"${opt_price:,.2f}", f"{opt_price - price:+.2f} vs current")
    m2.metric("Predicted Demand", f"{opt_demand:,.0f} units")
    m3.metric("Expected Revenue", f"${opt_revenue:,.0f}")
    m4.metric("Revenue Uplift", f"{uplift:+.1f}%")

    sim_df = pd.DataFrame(opt["simulation_table"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=sim_df["price"], y=sim_df["expected_revenue"],
        name="Revenue", mode="lines",
        line=dict(color="#68d391", width=3),
        fill="tozeroy", fillcolor="rgba(104,211,145,0.08)",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=sim_df["price"], y=sim_df["predicted_demand"],
        name="Demand", mode="lines",
        line=dict(color="#63b3ed", width=2, dash="dot"),
    ), secondary_y=True)
    fig.add_vline(x=opt_price, line_dash="dash", line_color="#f6ad55", line_width=2,
                  annotation_text=f"  Optimal ${opt_price:.2f}", annotation_font_color="#f6ad55")
    fig.add_vline(x=price, line_dash="dot", line_color="#a0aec0",
                  annotation_text=f"  Current ${price:.2f}", annotation_font_color="#a0aec0")
    fig.update_layout(**THEME, height=420, title="Revenue & Demand Across Simulated Prices",
                      xaxis_title="Price ($)", yaxis_title="Revenue ($)", yaxis2_title="Demand (units)",
                      legend=dict(x=0.01, y=0.98))
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### Top 10 Revenue-Maximizing Prices")
        top10 = pd.DataFrame(opt["top_10_prices"])
        top10.columns = ["Price ($)", "Demand (units)", "Revenue ($)", "vs Base (%)", "Is Optimal"]
        top10 = top10.drop(columns=["Is Optimal"])
        st.dataframe(top10.style.highlight_max(subset=["Revenue ($)"], color="#1a3728"), use_container_width=True)

    with col_r:
        st.markdown("#### Revenue Comparison")
        comparison = pd.DataFrame({
            "Scenario": ["Current Price", "Optimal Price", "Competitor Price"],
            "Price ($)": [price, opt_price, comp_price],
            "Revenue ($)": [
                opt["base_revenue"],
                opt_revenue,
                predict(comp_price, comp_price, promo, stock, season, holiday, weekday, model) * comp_price,
            ],
        })
        fig_bar = go.Figure(go.Bar(
            x=comparison["Scenario"],
            y=comparison["Revenue ($)"],
            marker_color=["#718096", "#68d391", "#fc8181"],
            text=[f"${v:,.0f}" for v in comparison["Revenue ($)"]],
            textposition="outside",
        ))
        fig_bar.update_layout(**THEME, height=300, showlegend=False, yaxis_title="Revenue ($)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Download Simulation Results")
    csv = sim_df.to_csv(index=False)
    st.download_button("⬇️ Download as CSV", data=csv, file_name="price_simulation.csv", mime="text/csv")


# ── TAB 3: Elasticity Analysis ─────────────────────────────────────────────────
with tab_elastic:
    st.markdown("### Price Elasticity of Demand Analysis")

    with st.spinner("Computing elasticity curve…"):
        from src.price_elasticity import generate_elasticity_report
        elas = generate_elasticity_report(price, comp_price, promo, stock, model,
                                          season=season, holiday=holiday, weekday=weekday)

    interp = elas["interpretation"]
    median_e = elas["median_elasticity"]

    e1, e2, e3 = st.columns(3)
    e1.metric("Median Elasticity", f"{median_e:.3f}")
    e2.metric("Category", interp["category"])
    e3.metric("Optimal Price (Elasticity)", f"${elas['optimal_price']:.2f}")

    st.markdown(f"""
    <div class="rec-box">
        <strong>{interp['category']}</strong> ({interp['sign_explanation']})<br/>
        {interp['action']}
    </div>
    """, unsafe_allow_html=True)

    curve = pd.DataFrame(elas["curve_data"])

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=curve["price"], y=curve["elasticity"].fillna(0),
            mode="lines+markers", line=dict(color="#f6ad55", width=2.5),
            name="Elasticity",
        ))
        fig_e.add_hline(y=-1, line_dash="dash", line_color="#fc8181",
                        annotation_text="Unit elastic (−1)")
        fig_e.add_hline(y=0, line_dash="dot", line_color="#718096")
        fig_e.update_layout(**THEME, height=340, title="Elasticity vs. Price",
                            xaxis_title="Price ($)", yaxis_title="Elasticity")
        st.plotly_chart(fig_e, use_container_width=True)

    with col_e2:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=curve["price"], y=curve["revenue"],
            mode="lines", fill="tozeroy",
            line=dict(color="#68d391", width=2.5),
            fillcolor="rgba(104,211,145,0.1)",
            name="Revenue",
        ))
        opt_idx = curve["revenue"].idxmax()
        fig_rev.add_vline(x=curve.iloc[opt_idx]["price"], line_dash="dash", line_color="#f6ad55",
                          annotation_text="  Max Revenue")
        fig_rev.update_layout(**THEME, height=340, title="Revenue Curve",
                              xaxis_title="Price ($)", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("#### Business Recommendations")
    for rec in elas["recommendations"]:
        st.markdown(f'<div class="rec-box">💡 {rec}</div>', unsafe_allow_html=True)

    st.markdown("#### Elasticity Formula")
    st.latex(r"\text{Elasticity} = \frac{\%\ \Delta\ \text{Demand}}{\%\ \Delta\ \text{Price}} = \frac{(Q_2 - Q_1)/\bar{Q}}{(P_2 - P_1)/\bar{P}}")


# ── TAB 4: Model Metrics ───────────────────────────────────────────────────────
with tab_metrics:
    st.markdown("### Model Performance Comparison")

    if metrics_summary and "all_metrics" in metrics_summary:
        all_m = metrics_summary["all_metrics"]
        rows = [{"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"], "R²": v["R2"]} for k, v in all_m.items()]
        df_metrics = pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)
        df_metrics["Best"] = df_metrics["Model"] == metrics_summary["best_model_name"]

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Metrics Table")
            st.dataframe(
                df_metrics[["Model", "MAE", "RMSE", "R²"]].style
                    .highlight_max(subset=["R²"], color="#1a3728")
                    .highlight_min(subset=["MAE", "RMSE"], color="#1a3728"),
                use_container_width=True,
            )

        with col_m2:
            st.markdown("#### R² Score Comparison")
            colors = ["#f6ad55" if r["Model"] == metrics_summary["best_model_name"] else "#63b3ed"
                      for _, r in df_metrics.iterrows()]
            fig_m = go.Figure(go.Bar(
                x=df_metrics["Model"], y=df_metrics["R²"],
                marker_color=colors,
                text=[f"{v:.4f}" for v in df_metrics["R²"]],
                textposition="outside",
            ))
            fig_m.update_layout(**THEME, height=280, showlegend=False,
                                yaxis_title="R² Score", yaxis_range=[0, 1.1])
            st.plotly_chart(fig_m, use_container_width=True)

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            fig_mae = go.Figure(go.Bar(
                x=df_metrics["Model"], y=df_metrics["MAE"],
                marker_color=["#fc8181" if r["Model"] == metrics_summary["best_model_name"] else "#718096"
                              for _, r in df_metrics.iterrows()],
                text=[f"{v:.1f}" for v in df_metrics["MAE"]], textposition="outside",
            ))
            fig_mae.update_layout(**THEME, height=260, showlegend=False,
                                  title="MAE (lower is better)", yaxis_title="MAE")
            st.plotly_chart(fig_mae, use_container_width=True)

        with col_m4:
            fig_rmse = go.Figure(go.Bar(
                x=df_metrics["Model"], y=df_metrics["RMSE"],
                marker_color=["#fc8181" if r["Model"] == metrics_summary["best_model_name"] else "#718096"
                              for _, r in df_metrics.iterrows()],
                text=[f"{v:.1f}" for v in df_metrics["RMSE"]], textposition="outside",
            ))
            fig_rmse.update_layout(**THEME, height=260, showlegend=False,
                                   title="RMSE (lower is better)", yaxis_title="RMSE")
            st.plotly_chart(fig_rmse, use_container_width=True)

        if metrics_summary.get("feature_importance"):
            st.markdown("#### Feature Importance")
            fi = metrics_summary["feature_importance"]
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h",
                marker_color="#63b3ed",
                text=[f"{v:.4f}" for v in fi_df["Importance"]],
                textposition="outside",
            ))
            fig_fi.update_layout(**THEME, height=350, title="Feature Importances (Best Model)",
                                 xaxis_title="Importance", margin=dict(l=140))
            st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.warning("Model metrics not available. Run the training pipeline first.")


# ── TAB 5: Data Exploration ────────────────────────────────────────────────────
with tab_eda:
    st.markdown("### Dataset Exploration")

    with st.spinner("Loading dataset statistics…"):
        eda_stats, df_raw = _load_eda()

    n_rows, n_cols = eda_stats["shape"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Samples", f"{n_rows:,}")
    d2.metric("Features", n_cols)
    d3.metric("Price-Demand Correlation", f"{eda_stats['price_demand_corr']:.4f}")
    d4.metric("Promotion Rate", f"{eda_stats['promotion_rate'] * 100:.1f}%")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### Demand Distribution")
        fig_hist = go.Figure(go.Histogram(
            x=df_raw["demand"], nbinsx=40,
            marker_color="#63b3ed", opacity=0.8,
        ))
        fig_hist.update_layout(**THEME, height=280, xaxis_title="Demand (units)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d2:
        st.markdown("#### Price vs Demand Scatter")
        sample = df_raw.sample(min(1000, len(df_raw)), random_state=42)
        fig_sc = go.Figure(go.Scatter(
            x=sample["price"], y=sample["demand"],
            mode="markers",
            marker=dict(color=sample["promotion"].map({0: "#63b3ed", 1: "#68d391"}),
                        size=5, opacity=0.6),
            text=sample["season"],
        ))
        fig_sc.update_layout(**THEME, height=280, xaxis_title="Price ($)", yaxis_title="Demand (units)")
        st.plotly_chart(fig_sc, use_container_width=True)

    col_d3, col_d4 = st.columns(2)

    with col_d3:
        st.markdown("#### Avg Demand by Season")
        by_season = eda_stats["demand_by_season"]
        fig_s = go.Figure(go.Bar(
            x=list(by_season.keys()), y=list(by_season.values()),
            marker_color=["#68d391", "#f6ad55", "#fc8181", "#76e4f7"],
            text=[f"{v:.0f}" for v in by_season.values()], textposition="outside",
        ))
        fig_s.update_layout(**THEME, height=260, showlegend=False, yaxis_title="Avg Demand")
        st.plotly_chart(fig_s, use_container_width=True)

    with col_d4:
        st.markdown("#### Price Distribution by Season")
        fig_box = go.Figure()
        for s, color in zip(["Spring", "Summer", "Fall", "Winter"],
                            ["#68d391", "#f6ad55", "#fc8181", "#76e4f7"]):
            data = df_raw[df_raw["season"] == s]["price"]
            fig_box.add_trace(go.Box(y=data, name=s, marker_color=color))
        fig_box.update_layout(**THEME, height=260, yaxis_title="Price ($)", showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### Sample Data Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    csv_data = df_raw.to_csv(index=False)
    st.download_button("⬇️ Download Full Dataset (CSV)", data=csv_data,
                       file_name="retail_sales.csv", mime="text/csv")


# ── TAB 6: Explainability (SHAP) ──────────────────────────────────────────────
with tab_shap:
    st.markdown("### Explainability — SHAP Feature Analysis")

    try:
        import shap

        @st.cache_data(show_spinner=False, ttl=600)
        def _compute_shap():
            from src.data_preprocessing import (
                load_raw_data, handle_missing_values, feature_engineering,
                encode_categoricals, scale_features, FEATURE_COLS
            )
            df = load_raw_data()
            df = handle_missing_values(df)
            df = feature_engineering(df)
            df, _ = encode_categoricals(df, fit=False)
            X = df[FEATURE_COLS]
            X_scaled, _ = scale_features(X, fit=False)
            sample_size = min(300, len(X_scaled))
            X_sample = X_scaled[:sample_size]
            mdl = load_model()
            if hasattr(mdl, "feature_importances_"):
                explainer = shap.TreeExplainer(mdl)
            else:
                explainer = shap.LinearExplainer(mdl, X_sample)
            sv = explainer.shap_values(X_sample)
            return sv, X_sample, FEATURE_COLS

        with st.spinner("Computing SHAP values (this may take a moment)…"):
            shap_values, X_sample, feat_cols = _compute_shap()

        st.success("SHAP analysis complete.")

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### Global Feature Importance (SHAP)")
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            fi_shap = pd.DataFrame({
                "Feature": feat_cols,
                "Mean |SHAP|": mean_abs_shap.round(4),
            }).sort_values("Mean |SHAP|", ascending=True)
            fig_shap = go.Figure(go.Bar(
                x=fi_shap["Mean |SHAP|"], y=fi_shap["Feature"],
                orientation="h",
                marker_color="#f6ad55",
                text=[f"{v:.4f}" for v in fi_shap["Mean |SHAP|"]],
                textposition="outside",
            ))
            fig_shap.update_layout(**THEME, height=350, xaxis_title="Mean |SHAP value|",
                                   margin=dict(l=140))
            st.plotly_chart(fig_shap, use_container_width=True)

        with col_s2:
            st.markdown("#### SHAP Value Distribution (Top Feature)")
            top_feat_idx = np.argmax(np.abs(shap_values).mean(axis=0))
            top_feat_name = feat_cols[top_feat_idx]
            fig_violin = go.Figure(go.Violin(
                y=shap_values[:, top_feat_idx],
                box_visible=True, meanline_visible=True,
                fillcolor="#63b3ed", line_color="#e2e8f0",
                opacity=0.7, name=top_feat_name,
            ))
            fig_violin.update_layout(**THEME, height=350, yaxis_title="SHAP Value",
                                     title=f"SHAP Distribution: {top_feat_name}")
            st.plotly_chart(fig_violin, use_container_width=True)

        st.markdown("#### SHAP Heatmap (Sample × Feature)")
        heatmap_data = shap_values[:50]
        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_data,
            x=feat_cols,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="SHAP"),
        ))
        fig_heat.update_layout(**THEME, height=400,
                               title="SHAP Values — First 50 Samples",
                               xaxis_title="Feature", yaxis_title="Sample Index")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Single Prediction Explanation")
        st.markdown("SHAP contribution for the **current sidebar inputs**:")
        from src.data_preprocessing import preprocess_single, FEATURE_COLS as FC
        X_single = preprocess_single(price, comp_price, promo, stock, season, holiday, weekday)
        mdl = load_model()
        if hasattr(mdl, "feature_importances_"):
            exp_single = shap.TreeExplainer(mdl)
        else:
            exp_single = shap.LinearExplainer(mdl, X_sample)
        sv_single = exp_single.shap_values(X_single)[0]
        shap_df = pd.DataFrame({
            "Feature": FC,
            "SHAP Value": sv_single.round(4),
            "Direction": ["⬆️ increases demand" if v > 0 else "⬇️ decreases demand" for v in sv_single],
        }).sort_values("SHAP Value", key=abs, ascending=False)
        st.dataframe(shap_df, use_container_width=True)

    except Exception as e:
        st.warning(f"SHAP analysis unavailable: {e}")
        st.info("Install `shap` and ensure the model is trained to enable this tab.")


# ── TAB 7: Retrain ─────────────────────────────────────────────────────────────
with tab_retrain:
    import json as _json
    import time as _time

    st.markdown("### ⚙️ Model Retraining Pipeline")
    st.markdown(
        "Retrain all three models (Linear Regression, Random Forest, XGBoost) on fresh data. "
        "The best model is automatically selected by R² and hot-swapped into production."
    )

    RETRAIN_HISTORY_PATH = os.path.join(BASE_DIR, "models", "retrain_history.json")
    REQUIRED_COLS = ["price", "competitor_price", "promotion", "inventory_level",
                     "season", "holiday", "weekday", "demand"]

    # ── Data source ──────────────────────────────────────────────────────────
    st.markdown("#### 1️⃣ Choose Data Source")
    data_source = st.radio(
        "Training data",
        ["Use existing dataset on disk", "Upload a new CSV"],
        horizontal=True,
    )

    uploaded_df = None
    if data_source == "Upload a new CSV":
        st.markdown(
            f"**Required columns:** `{'`, `'.join(REQUIRED_COLS)}`\n\n"
            "Season must be one of: `Spring`, `Summer`, `Fall`, `Winter`. "
            "`promotion` and `holiday` must be 0 or 1."
        )
        uploaded_file = st.file_uploader("Upload retail sales CSV", type=["csv"])
        if uploaded_file:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                missing_cols = set(REQUIRED_COLS) - set(uploaded_df.columns)
                if missing_cols:
                    st.error(f"Missing required columns: `{', '.join(sorted(missing_cols))}`")
                    uploaded_df = None
                else:
                    st.success(f"✅ Valid CSV — {len(uploaded_df):,} rows, {len(uploaded_df.columns)} columns")
                    st.dataframe(uploaded_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                uploaded_df = None

        # Download a sample template
        sample_data = pd.DataFrame({
            "price": [100.0, 120.0, 80.0],
            "competitor_price": [95.0, 115.0, 85.0],
            "promotion": [0, 1, 0],
            "inventory_level": [500, 300, 800],
            "season": ["Summer", "Winter", "Spring"],
            "holiday": [0, 1, 0],
            "weekday": [2, 5, 1],
            "demand": [1200, 950, 1400],
        })
        st.download_button(
            "⬇️ Download CSV template",
            data=sample_data.to_csv(index=False),
            file_name="retrain_template.csv",
            mime="text/csv",
        )

    # ── Trigger ───────────────────────────────────────────────────────────────
    st.markdown("#### 2️⃣ Run Retraining")

    can_train = (data_source == "Use existing dataset on disk") or (uploaded_df is not None)
    retrain_btn = st.button(
        "🚀 Start Retraining",
        disabled=not can_train,
        type="primary",
        use_container_width=True,
    )

    if retrain_btn:
        metrics_before = metrics_summary.get("all_metrics", {}) if metrics_summary else {}

        with st.spinner("Training Linear Regression, Random Forest, XGBoost… (~30 seconds)"):
            try:
                # Save uploaded CSV if provided
                if uploaded_df is not None:
                    raw_path = os.path.join(BASE_DIR, "data", "retail_sales.csv")
                    uploaded_df.to_csv(raw_path, index=False)
                    st.info(f"Saved {len(uploaded_df):,} rows to training dataset.")

                # Clear Streamlit cache so new model is loaded fresh
                st.cache_resource.clear()
                st.cache_data.clear()

                from src.model_training import run_training_pipeline
                summary_new = run_training_pipeline()

                # Append to history file
                entry = {
                    "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                    "source": "dashboard_upload" if uploaded_df is not None else "existing_data",
                    "best_model": summary_new["best_model_name"],
                    "metrics_before": metrics_before,
                    "metrics_after": summary_new["all_metrics"],
                }
                os.makedirs(os.path.dirname(RETRAIN_HISTORY_PATH), exist_ok=True)
                history = []
                if os.path.exists(RETRAIN_HISTORY_PATH):
                    with open(RETRAIN_HISTORY_PATH) as f:
                        history = _json.load(f)
                history.append(entry)
                with open(RETRAIN_HISTORY_PATH, "w") as f:
                    _json.dump(history[-50:], f, indent=2)

                st.success(f"✅ Retraining complete! Best model: **{summary_new['best_model_name']}**")

                # ── Before / After comparison ──────────────────────────────
                if metrics_before:
                    st.markdown("#### Before vs. After Metrics")
                    rows = []
                    for model_name, after_m in summary_new["all_metrics"].items():
                        before_m = metrics_before.get(model_name, {})
                        rows.append({
                            "Model": model_name,
                            "R² Before": before_m.get("R2", "—"),
                            "R² After": after_m["R2"],
                            "MAE Before": before_m.get("MAE", "—"),
                            "MAE After": after_m["MAE"],
                            "RMSE Before": before_m.get("RMSE", "—"),
                            "RMSE After": after_m["RMSE"],
                        })
                    df_compare = pd.DataFrame(rows)
                    st.dataframe(df_compare, use_container_width=True)

                    # R² bar chart
                    fig_ba = go.Figure()
                    for model_name in summary_new["all_metrics"]:
                        before_r2 = metrics_before.get(model_name, {}).get("R2", 0)
                        after_r2 = summary_new["all_metrics"][model_name]["R2"]
                        fig_ba.add_trace(go.Bar(name=f"{model_name} Before",
                                                x=[model_name], y=[before_r2],
                                                marker_color="#718096", opacity=0.7))
                        fig_ba.add_trace(go.Bar(name=f"{model_name} After",
                                                x=[model_name], y=[after_r2],
                                                marker_color="#68d391"))
                    fig_ba.update_layout(**THEME, barmode="group", height=320,
                                         title="R² Score: Before vs. After",
                                         yaxis_title="R²", yaxis_range=[0, 1.05],
                                         showlegend=True)
                    st.plotly_chart(fig_ba, use_container_width=True)

                st.info("Refresh the page to see updated model metrics across all tabs.")

            except Exception as e:
                st.error(f"Retraining failed: {e}")

    # ── History ───────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Retraining History")

    if os.path.exists(RETRAIN_HISTORY_PATH):
        with open(RETRAIN_HISTORY_PATH) as f:
            history_data = _json.load(f)

        if history_data:
            history_rows = []
            for run in reversed(history_data):
                after = run.get("metrics_after", {})
                best = run.get("best_model", "—")
                best_r2 = after.get(best, {}).get("R2", "—") if isinstance(after.get(best), dict) else "—"
                history_rows.append({
                    "Timestamp (UTC)": run.get("timestamp", "—")[:19].replace("T", " "),
                    "Data Source": run.get("source", "—"),
                    "Best Model": best,
                    "Best R²": best_r2,
                })
            df_hist = pd.DataFrame(history_rows)
            st.dataframe(df_hist, use_container_width=True)

            # Download history
            st.download_button(
                "⬇️ Download Retrain History (CSV)",
                data=df_hist.to_csv(index=False),
                file_name="retrain_history.csv",
                mime="text/csv",
            )
        else:
            st.info("No retraining runs recorded yet.")
    else:
        st.info("No retraining history found. Run your first retrain above.")

    # ── API reference ─────────────────────────────────────────────────────────
    with st.expander("🔌 API Usage (cURL)"):
        st.code("""# Retrain on existing data
curl -X POST http://localhost:8000/retrain

# Retrain with new CSV
curl -X POST http://localhost:8000/retrain \\
  -F "file=@/path/to/new_data.csv"

# Check training status
curl http://localhost:8000/retrain-status

# View history
curl http://localhost:8000/retrain-history""", language="bash")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#4a5568; font-size:0.8rem;'>Price Optimization Engine · "
    "ML-powered pricing intelligence · Built with Streamlit + Scikit-Learn + XGBoost</center>",
    unsafe_allow_html=True,
)
