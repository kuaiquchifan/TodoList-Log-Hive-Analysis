# 09_viz_backend.py
import io
import base64
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- CONFIG ----------
PARQUET_ACTIVE = "output_aggregate_further_data/01_dws_user_active_1d.parquet"
PARQUET_DATE = "output_data/02_dim_date.parquet"
PARQUET_NEWOLD = "output_aggregate_further_data/07_dws_new_old_user_summary_by_register_date.parquet"
PARQUET_BEHAVIOR = "output_aggregate_further_data/02_dws_user_behavior_summary.parquet"
PARQUET_USER = "output_data/01_dim_user.parquet"
PARQUET_RETAIN = "output_aggregate_further_data/08_dws_user_retain_summary_by_register_date.parquet"
PARQUET_CHANNEL = "output_aggregate_further_data/06_dws_channel_region_summary.parquet"

# 读文件并缓存（模块加载时）
def try_read(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

DF_ACTIVE = try_read(PARQUET_ACTIVE)
DF_DATE = try_read(PARQUET_DATE)
DF_NEWOLD = try_read(PARQUET_NEWOLD)
DF_BEHAVIOR = try_read(PARQUET_BEHAVIOR)
DF_USER = try_read(PARQUET_USER)
DF_RETAIN = try_read(PARQUET_RETAIN)
DF_CHANNEL = try_read(PARQUET_CHANNEL)

# ---------- HELPERS ----------
def to_date_series(s):
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return pd.to_datetime(s.astype(int).astype(str), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")

def _make_kpi_card(title, value, color="primary"):
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            display = f"{value:,}"
        else:
            display = str(value)
    except Exception:
        display = str(value)
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted"),
            html.H4(display, className="mt-2")
        ]),
        color=color, inverse=False, className="m-1"
    )




# ---------- BUILDERS ----------
def _build_active_inner(df_active):
    df = df_active.copy()
    if "stat_date" not in df.columns:
        return go.Figure(), {}, None
    df["date"] = to_date_series(df["stat_date"])
    df = df.dropna(subset=["date"])
    if df.empty:
        return go.Figure(), {}, None
    dau = df.groupby("date")["user_id"].nunique().reset_index().rename(columns={"user_id": "dau"}).sort_values("date")
    daily_users = df.groupby("date")["user_id"].apply(lambda s: set(s.tolist()))
    all_dates = pd.date_range(dau["date"].min(), dau["date"].max(), freq="D")
    res = pd.DataFrame(index=all_dates).join(dau.set_index("date")["dau"]).fillna(0)
    res = res.reset_index().rename(columns={"index": "date"})
    def unique_count_over_window(end_date, days):
        window = pd.date_range(end_date - pd.Timedelta(days=days - 1), end_date, freq="D")
        users = set()
        for d in window:
            if d in daily_users.index:
                users |= daily_users.loc[d]
        return len(users)
    res["wau"] = [unique_count_over_window(d.date(), 7) for d in res["date"]]
    res["mau"] = [unique_count_over_window(d.date(), 30) for d in res["date"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["date"], y=res["dau"], mode="lines+markers", name="DAU"))
    fig.add_trace(go.Scatter(x=res["date"], y=res["wau"], mode="lines+markers", name="WAU"))
    fig.add_trace(go.Scatter(x=res["date"], y=res["mau"], mode="lines+markers", name="MAU"))
    fig.update_layout(title="用户活跃趋势", xaxis_title="日期", yaxis_title="用户数", hovermode="x unified", template="plotly_white")
    last_date = res["date"].max()
    last_row = res[res["date"] == last_date].iloc[0]
    cards = {
        "DAU": {"today": int(last_row.get("dau", 0)), "7d_avg": int(res["dau"].tail(7).mean()), "30d_avg": int(res["dau"].tail(30).mean())},
        "WAU": {"today": int(last_row.get("wau", 0)), "7d_avg": int(res["wau"].tail(7).mean()), "30d_avg": int(res["wau"].tail(30).mean())},
        "MAU": {"today": int(last_row.get("mau", 0)), "7d_avg": int(res["mau"].tail(7).mean()), "30d_avg": int(res["mau"].tail(30).mean())},
    }
    return fig, cards, last_date

def get_newold_df():
    df = DF_NEWOLD.copy()
    if "stat_date" in df.columns:
        df["date"] = to_date_series(df["stat_date"])
    if "date" in df.columns:
        df = df.dropna(subset=["date"]).sort_values("date")
    return df

def build_newold_stack(df_newold=None):
    df = get_newold_df() if df_newold is None else df_newold.copy()
    if "date" not in df.columns and "stat_date" in df.columns:
        df["date"] = to_date_series(df["stat_date"])
    if "date" in df.columns:
        df = df.dropna(subset=["date"]).sort_values("date")

    fig = go.Figure()
    if {"date", "new_user_cnt", "old_user_cnt"}.issubset(df.columns):
        fig.add_trace(go.Bar(x=df["date"], y=df["new_user_cnt"], name="新用户"))
        fig.add_trace(go.Bar(x=df["date"], y=df["old_user_cnt"], name="老用户"))
        fig.update_layout(
            barmode="stack",
            title="新/老用户结构（堆叠）",
            xaxis_title="日期",
            yaxis_title="用户数",
            template="plotly_white"
        )
    return df, fig



def build_extra_figs(df_behavior, df_retain, df_channel):
    freq_pie, freq_bar = go.Figure(), go.Figure()
    retain_line, retain_heat = go.Figure(), go.Figure()
    channel_bar, channel_scatter = go.Figure(), go.Figure()

    # --- 频次 ---
    try:
        if not df_behavior.empty:
            dfb = df_behavior.copy()
            active_days = pd.to_numeric(DF_BEHAVIOR["active_day_cnt"], errors="coerce")
            print(active_days.describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

            active_days_only = active_days[active_days > 0]

            q25 = active_days_only.quantile(0.25)
            q75 = active_days_only.quantile(0.75)
            print(f"Q25: {q25}, Q75: {q75}")
            if {"user_id", "active_day_cnt"}.issubset(dfb.columns):
                def bucket(value):
                    days = pd.to_numeric(value, errors="coerce")
                    if pd.isna(days) or days <= 0:
                        return "无活跃"
                    if days <= q25:
                        return "低频"
                    if days <= q75:
                        return "中频"
                    return "高频"

                # 从用户维表取全体用户；没有对应行为记录的用户按 0 个活跃日处理
                behavior_users = dfb[["user_id", "active_day_cnt"]].drop_duplicates("user_id")
                if not DF_USER.empty and "user_id" in DF_USER.columns:
                    user_freq = DF_USER[["user_id"]].drop_duplicates().merge(
                        behavior_users,
                        on="user_id",
                        how="left"
                    )
                else:
                    # 若维表不可用，只统计行为汇总表中出现过的用户
                    user_freq = behavior_users.copy()

                user_freq["active_day_cnt"] = pd.to_numeric(
                    user_freq["active_day_cnt"], errors="coerce"
                ).fillna(0)
                user_freq["freq_bucket"] = user_freq["active_day_cnt"].apply(bucket)

                print(
                    user_freq["freq_bucket"].value_counts().reindex(
                        ["无活跃", "低频", "中频", "高频"],
                        fill_value=0
                    )
                )

                bucket_order = ["无活跃", "低频", "中频", "高频"]
                freq_agg = (
                    user_freq.groupby("freq_bucket")["user_id"]
                    .nunique()
                    .reindex(bucket_order, fill_value=0)
                    .rename("cnt")
                    .reset_index()
                )

                total_users = user_freq["user_id"].nunique()
                active_users = user_freq.loc[
                    user_freq["active_day_cnt"] > 0, "user_id"
                ].nunique()

                pie_df = pd.DataFrame({
                    "label": ["活跃用户", "非活跃用户"],
                    "cnt": [active_users, max(total_users - active_users, 0)]
                })
                freq_pie = px.pie(
                    pie_df,
                    names="label",
                    values="cnt",
                    title="全周期用户活跃占比（活跃 vs 非活跃）",
                    hole=0.3
                )

                freq_bar = px.bar(
                    freq_agg,
                    x="freq_bucket",
                    y="cnt",
                    category_orders={"freq_bucket": bucket_order},
                    title="全周期用户活跃频次分层"
                )
    except Exception:
        pass  


    # --- 留存 ---
    try:
        if not df_retain.empty:
            dfr = df_retain.copy()
            if "first_active_date" in dfr.columns:
                dfr["date"] = to_date_series(dfr["first_active_date"])
            dfr = dfr.dropna(subset=["date"]).sort_values("date") if "date" in dfr.columns else dfr

            # 若没有 retain_1d/7d/30d，用 total_new_users 与 dayX_retained_users 计算
            if not {"retain_1d", "retain_7d", "retain_30d"}.issubset(dfr.columns):
                if "total_new_users" in dfr.columns:
                    if "day1_retained_users" in dfr.columns:
                        dfr["retain_1d"] = dfr["day1_retained_users"] / dfr["total_new_users"].replace({0: np.nan})
                    if "day7_retained_users" in dfr.columns:
                        dfr["retain_7d"] = dfr["day7_retained_users"] / dfr["total_new_users"].replace({0: np.nan})
                    if "day30_retained_us" in dfr.columns:
                        dfr["retain_30d"] = dfr["day30_retained_us"] / dfr["total_new_users"].replace({0: np.nan})
                    elif "day30_retained_users" in dfr.columns:
                        dfr["retain_30d"] = dfr["day30_retained_users"] / dfr["total_new_users"].replace({0: np.nan})

            if {"retain_1d", "retain_7d", "retain_30d"}.issubset(dfr.columns):
                dfm = dfr[["date", "retain_1d", "retain_7d", "retain_30d"]].melt(id_vars="date", var_name="period", value_name="rate")
                retain_line = px.line(dfm, x="date", y="rate", color="period", title="次日/7日/30日留存曲线")
                retain_line.update_yaxes(tickformat=".0%")

            if {"cohort_date", "day_index", "retain_rate"}.issubset(dfr.columns):
                pivot = dfr.pivot_table(index="cohort_date", columns="day_index", values="retain_rate")
                if not pivot.empty:
                    retain_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Viridis"))
                    retain_heat.update_layout(title="留存热力图", xaxis_title="留存天数", yaxis_title="注册日期")
    except Exception:
        pass

    # --- 渠道质量 ---
    try:
        if not df_channel.empty:
            chc = df_channel.copy()
            channel_col = next((c for c in ("register_channel", "channel", "reg_channel") if c in chc.columns), None)
            new_cnt_col = next((c for c in ("new_user_cnt", "new_users", "new_register_cnt", "register_cnt", "new_registers", "active_user_cnt", "event_cnt") if c in chc.columns), None)

            if channel_col and new_cnt_col:
                # 规范字符串
                if chc[channel_col].dtype == object:
                    chc[channel_col] = chc[channel_col].astype(str).str.strip()
                ch_renamed = chc.rename(columns={channel_col: "channel", new_cnt_col: "new_user_cnt"})
                reg = ch_renamed.groupby("channel")["new_user_cnt"].sum().reset_index()

                # 若 df_retain 含 channel 与 retain_7d，则合并
                if "channel" in DF_RETAIN.columns and "retain_7d" in DF_RETAIN.columns:
                    retain_by_channel = DF_RETAIN.groupby("channel")["retain_7d"].mean().reset_index()
                    merged = reg.merge(retain_by_channel, on="channel", how="left")
                else:
                    merged = reg

                if "province" in chc.columns or "city" in chc.columns:
                    region_col = "province" if "province" in chc.columns else "city"
                    ch_region = ch_renamed.groupby(["channel", region_col])["new_user_cnt"].sum().reset_index()
                    channel_bar = px.bar(ch_region, x="channel", y="new_user_cnt", color=region_col, barmode="group", title="渠道-区域 新用户数（或活跃数）")
                else:
                    channel_bar = px.bar(merged, x="channel", y="new_user_cnt", title="渠道 新用户数（或活跃数）")

                if "retain_7d" in merged.columns:
                    channel_scatter = px.scatter(merged, x="new_user_cnt", y="retain_7d", text="channel", title="注册量 vs 7日留存")
                    channel_scatter.update_yaxes(tickformat=".0%")
    except Exception:
        pass

    return (freq_pie, freq_bar), (retain_line, retain_heat), (channel_bar, channel_scatter)

# ---------- RENDERERS: 返回界面片段 ----------
def render_overview():
    fig_line, cards, last_date = _build_active_inner(DF_ACTIVE)
    df_newold, fig_stack = build_newold_stack(DF_NEWOLD)
    (freq_pie, freq_bar), (retain_line, retain_heat), (channel_bar, channel_scatter) = build_extra_figs(DF_BEHAVIOR, DF_RETAIN, DF_CHANNEL)

    # 下拉选项
    date_options = []
    try:
        date_options = [{"label": d.strftime("%Y-%m-%d"), "value": d.strftime("%Y-%m-%d")} for d in sorted(df_newold["date"].dropna().unique())]
    except Exception:
        date_options = []

    default_date = None
    try:
        default_date = df_newold["date"].max().strftime("%Y-%m-%d")
    except Exception:
        default_date = None

    rows = [
            dbc.Row(
            [
                dbc.Col(_make_kpi_card("DAU", cards.get("DAU", {}).get("today", 0),
                                       color="primary"), width=2),
                dbc.Col(_make_kpi_card("WAU", cards.get("WAU", {}).get("today", 0),
                                       color="info"), width=2),
                dbc.Col(_make_kpi_card("MAU", cards.get("MAU", {}).get("today", 0),
                                       color="success"), width=2),
            ],
            className="mb-3"
        ),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_line), width=12)], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_stack), width=8),
            dbc.Col([
                dbc.Row([dbc.Col(html.Div([html.Label("选择日期："),
                                           dcc.Dropdown(id="newold-date", options=date_options, value=default_date, clearable=False, style={"width": "220px"})]),
                                 width=12)], className="mb-2"),
                dbc.Row([dbc.Col(dcc.Graph(id="newold-pie"), width=12)])
            ], width=4)
        ], className="mb-3"),
        dbc.Row([dbc.Col(dcc.Graph(figure=freq_pie), width=6), dbc.Col(dcc.Graph(figure=freq_bar), width=6)], className="mb-4"),
        dbc.Row([dbc.Col(dcc.Graph(figure=retain_line), width=6), dbc.Col(dcc.Graph(figure=channel_bar), width=6)], className="mb-4"),

    ]

    # 回调仍需由主 app 注册：提供 newold-pie 的回调函数工厂（返回回调函数和依赖）
    # 这里返回布局片段；主文件的回调使用 DF_NEWOLD（模块级缓存）来生成饼图
    return rows


def newold_pie_figure(selected_date):
    if not selected_date:
        return go.Figure()

    df = get_newold_df()
    if "date" not in df.columns:
        return go.Figure()

    row = df[df["date"].dt.strftime("%Y-%m-%d") == selected_date]
    if row.empty:
        return go.Figure()

    row = row.iloc[0]
    new_cnt = int(float(row.get("new_user_cnt", 0)))
    old_cnt = int(float(row.get("old_user_cnt", 0)))

    fig = px.pie(
        names=["新用户", "老用户"],
        values=[new_cnt, old_cnt],
        title=f"{selected_date} 新/老用户占比"
    )
    fig.update_layout(
        title={"text": f"{selected_date} 新/老用户占比", "x": 0.5, "xanchor": "center"},
        template="plotly_white"
    )
    fig.update_traces(textinfo="percent+label")
    return fig