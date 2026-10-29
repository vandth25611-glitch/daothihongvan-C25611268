# -*- coding: utf-8 -*-
"""
DASHBOARD QUẢN TRỊ TỒN KHO DỰA TRÊN DỰ BÁO XÁC SUẤT
Chạy:  streamlit run app.py
Ứng dụng chỉ đọc các file kết quả đã tính sẵn trong thư mục data/ (xuất từ notebook),
sau đó tính lại toàn bộ quyết định mỗi khi người dùng thay đổi tham số ở thanh bên.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from engine import Params, decide, transfers, summary, weekly_digest, horizon_quantile, QCOLS

DATA = Path(__file__).parent / "data"
BLUE, ORANGE, AQUA, YELLOW, RED, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#52514e", "#e6e5e1"
DIV_COLOR = {"Scholar Footwear": BLUE, "Femme Footwear": ORANGE, "Junior Apparel": AQUA}
NOW = pd.Timestamp("2026-04-20")

st.set_page_config(page_title="Quản trị tồn kho theo dự báo xác suất", layout="wide")


@st.cache_data
def load():
    return {k: pd.read_parquet(DATA / f"{k}.parquet") for k in
            ["sku_now", "store_stock", "history", "daily_stock", "weekly_division"]} | {
            "eval": pd.read_csv(DATA / "eval_by_model_H.csv")}


def style(fig, h=320, ytitle=None, xtitle=None):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(size=12, color="#0b0b0b"), legend=dict(orientation="h", y=1.12, x=0),
                      hovermode="x unified")
    fig.update_xaxes(showgrid=False, linecolor=GRID, title=xtitle)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle, tickformat=",")
    return fig


def money(v):
    return f"{v:,.0f}"


D = load()

# ------------------------------------------------------------------ THANH BÊN: THAM SỐ
# Có thể đặt sẵn kịch bản qua đường dẫn, ví dụ:  ?lead=5&hold=30&rho=0.56&cover=8
def qp(name, default, cast=float):
    try:
        return cast(st.query_params.get(name, default))
    except (TypeError, ValueError):
        return default

st.sidebar.header("Tham số mô phỏng")
st.sidebar.caption("Thay đổi tham số, mọi con số và đề xuất bên phải được tính lại ngay.")
with st.sidebar.expander("Cung ứng", expanded=True):
    lead = st.slider("Thời gian giao hàng của nhà cung cấp (tuần)", 1, 6, qp("lead", 3, int),
                     help="Dữ liệu không có ngày đặt hàng nên đây là giả định. Trung vị thời gian từ lúc cửa hàng hết hàng đến khi được bổ sung trong dữ liệu là 19 ngày.")
    review = st.slider("Chu kỳ rà soát đặt hàng (tuần)", 1, 2, 1)
with st.sidebar.expander("Tài chính", expanded=True):
    hold = st.slider("Chi phí vốn lưu kho (%/năm)", 5, 40, qp("hold", 20, int)) / 100
    hold_weeks = st.slider("Số tuần hàng dư nằm kho trước khi xả", 4, 40, 20)
    goodwill = st.slider("Tổn thất uy tín khi thiếu hàng (% giá bán)", 0, 30, 0) / 100
    rho_mode = st.radio("Tỷ lệ thu hồi vốn khi xả hàng", ["Theo dữ liệu từng nhóm hàng", "Tự đặt một mức chung"],
                        index=1 if "rho" in st.query_params else 0)
    rho_override = -1.0
    if rho_mode == "Tự đặt một mức chung":
        rho_override = st.slider("Tỷ lệ thu hồi (giá xả / giá vốn)", 0.0, 1.0, qp("rho", 0.56),
                                 help="0.56 là mức thực tế của hàng bán ở trạng thái Xả hàng trong dữ liệu.")
with st.sidebar.expander("Mức phục vụ", expanded=False):
    use_override = st.checkbox("Bỏ qua q*, dùng một mức phục vụ chung")
    service = st.slider("Mức phục vụ chung", 0.50, 0.95, 0.90, 0.05) if use_override else -1.0
with st.sidebar.expander("Ngưỡng hàng dư và giảm giá", expanded=False):
    target_cover = st.slider("Số tuần tồn mục tiêu", 4, 26, qp("cover", 12, int))
    md1 = st.slider("Giảm giá cấp 1 khi tồn vượt (tuần)", 12, 52, 26)
    md2 = st.slider("Giảm giá cấp 2 khi tồn vượt (tuần)", md1, 78, max(39, md1))
    clr = st.slider("Xả hàng khi tồn vượt (tuần)", md2, 104, max(52, md2))
    tr_min = st.slider("Điều chuyển: cửa hàng nhận phải bán tối thiểu (đơn vị/8 tuần)", 1, 5, 2)

P = Params(lead_time=lead, review=review, holding_rate=hold, hold_weeks=hold_weeks, goodwill=goodwill,
           rho_override=rho_override, service_override=service, target_cover=target_cover,
           md1_cover=md1, md2_cover=md2, clear_cover=clr, tr_min_sales=tr_min)


@st.cache_data
def compute(p: Params):
    dec = decide(D["sku_now"], p)
    tr = transfers(dec, D["store_stock"], p)
    return dec, tr, summary(dec, tr, p)


dec, tr, S = compute(P)
_, _, S0 = compute(Params())          # kịch bản cơ sở để so sánh

st.title("Quản trị tồn kho theo dự báo xác suất")
st.caption(f"Chuỗi bán lẻ giày dép 40 cửa hàng · dữ liệu đến 19/04/2026 · thời điểm ra quyết định: {NOW:%d/%m/%Y} · "
           f"mô hình dự báo: LightGBM hồi quy phân vị · đơn vị tiền theo dữ liệu gốc")

tabs = st.tabs(["Tổng quan", "Dự báo", "Nhập hàng", "Hàng dư và xả hàng", "Điều chuyển", "So sánh kịch bản", "Bản tin tự động"])

# ------------------------------------------------------------------ 1. TỔNG QUAN
with tabs[0]:
    c = st.columns(5)
    def diff(x, fmt):                       # chỉ hiện chênh lệch khi kịch bản khác kịch bản cơ sở
        return None if abs(x) < 1e-9 else f"{x:{fmt}} so với cơ sở"
    c[0].metric("Giá trị tồn kho", money(S["stock_value"])); c[0].caption(f"{S['stock_units']:,.0f} đơn vị · {S['n_sku']:,} mã hàng")
    c[1].metric("Vốn trong hàng dư", money(S["excess_value"]), diff(S["excess_value"] - S0["excess_value"], "+,.0f"), delta_color="inverse")
    c[1].caption(f"{S['excess_share']:.0%} giá trị tồn")
    c[2].metric("Chi phí vốn / năm", money(S["holding_cost_year"]), diff(S["holding_cost_year"] - S0["holding_cost_year"], "+,.0f"), delta_color="inverse")
    c[2].caption("của phần hàng dư")
    c[3].metric("Mã hàng cần nhập", f"{S['order_skus']:,}", diff(S["order_skus"] - S0["order_skus"], "+,"), delta_color="off")
    c[3].caption(f"{S['order_units']:,.0f} đơn vị · {money(S['order_value'])} giá vốn")
    c[4].metric("q* bình quân", f"{S['q_star_mean']:.2f}", diff(S["q_star_mean"] - S0["q_star_mean"], "+.2f"), delta_color="off")
    c[4].caption("mức phục vụ tối ưu")

    # Cảnh báo
    if S["excess_share"] > 0.40:
        st.error(f"⛔ Cảnh báo chôn vốn: {S['excess_share']:.0%} giá trị tồn kho là hàng dư so với mức tồn mục tiêu "
                 f"{P.target_cover:.0f} tuần. Chi phí vốn ước tính {money(S['holding_cost_year'])}/năm; "
                 f"nếu phải xả toàn bộ, khoản lỗ dự kiến {money(S['markdown_loss'])}.")
    elif S["excess_share"] > 0.25:
        st.warning(f"⚠️ Hàng dư chiếm {S['excess_share']:.0%} giá trị tồn kho ({money(S['excess_value'])}). "
                   f"Chi phí vốn ước tính {money(S['holding_cost_year'])}/năm; lỗ dự kiến nếu xả: {money(S['markdown_loss'])}.")
    else:
        st.success(f"✅ Hàng dư ở mức {S['excess_share']:.0%} giá trị tồn kho, trong ngưỡng kiểm soát.")
    a = S["actions"]
    n_md = a.get("Giảm giá cấp 1", 0) + a.get("Giảm giá cấp 2", 0) + a.get("Xả hàng", 0)
    if S["order_skus"] > 0:
        st.warning(f"⚠️ Nguy cơ thiếu hàng: {S['order_skus']} mã hàng có tồn thấp hơn mức mục tiêu cho "
                   f"{P.protect()} tuần tới; cần nhập {S['order_units']:,.0f} đơn vị ({money(S['order_value'])} giá vốn).")
    st.info(f"ℹ️ {n_md} mã hàng được đề xuất giảm giá hoặc xả · {S['transfer_moves']:,} dòng điều chuyển "
            f"({S['transfer_units']:,.0f} đơn vị) giữa các cửa hàng.")

    l, r = st.columns(2)
    with l:
        st.subheader("Giá trị tồn kho toàn chuỗi theo ngày")
        ds = D["daily_stock"]
        fig = go.Figure(go.Scatter(x=ds["date"], y=ds["cost"], mode="lines", line=dict(color=BLUE, width=2),
                                   fill="tozeroy", fillcolor="rgba(42,120,214,0.10)", name="Giá trị tồn (giá vốn)"))
        st.plotly_chart(style(fig, ytitle="Giá vốn"), width="stretch")
    with r:
        st.subheader("Sản lượng bán theo tuần")
        wd = D["weekly_division"]
        fig = go.Figure()
        for dv, col in DIV_COLOR.items():
            x = wd[wd["division"] == dv]
            fig.add_trace(go.Scatter(x=x["week_start"], y=x["y"], mode="lines", name=dv, line=dict(color=col, width=2)))
        st.plotly_chart(style(fig, ytitle="Đơn vị / tuần"), width="stretch")

    st.subheader("Tồn kho và hàng dư theo nhóm hàng")
    cg = dec.groupby("category").agg(stock=("stock_value", "sum"), excess=("excess_value", "sum")).sort_values("stock").tail(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=cg.index, x=cg["stock"] - cg["excess"], orientation="h", name="Trong mức mục tiêu", marker_color=BLUE))
    fig.add_trace(go.Bar(y=cg.index, x=cg["excess"], orientation="h", name="Hàng dư", marker_color=ORANGE))
    fig.update_layout(barmode="stack", bargap=0.35)
    st.plotly_chart(style(fig, 460, xtitle="Giá trị tồn (giá vốn)").update_layout(hovermode="y unified"), width="stretch")

# ------------------------------------------------------------------ 2. DỰ BÁO
with tabs[1]:
    st.subheader("Dải dự báo xác suất theo mã hàng")
    c1, c2, c3 = st.columns([1, 1, 2])
    dv = c1.selectbox("Ngành hàng", ["Tất cả"] + sorted(dec["division"].unique()))
    pool = dec if dv == "Tất cả" else dec[dec["division"] == dv]
    ct = c2.selectbox("Nhóm hàng", ["Tất cả"] + sorted(pool["category"].unique()))
    pool = pool if ct == "Tất cả" else pool[pool["category"] == ct]
    pool = pool.sort_values("sales_8w", ascending=False)
    sku = c3.selectbox("Mã hàng (xếp theo sản lượng 8 tuần gần nhất)", pool["sku"].tolist(),
                       format_func=lambda s: f"{s} · bán 8 tuần: {dec.set_index('sku').at[s, 'sales_8w']:.0f} · tồn: {dec.set_index('sku').at[s, 'onhand']:.0f}")
    row = dec[dec["sku"] == sku].iloc[0]
    h = D["history"][D["history"]["sku"] == sku].sort_values("week_start")
    hz = [1, 2, 3, 4, 8]
    l, r = st.columns([3, 2])
    with l:
        g1, g2 = st.columns(2)
        fig = go.Figure(go.Scatter(x=h["week_start"], y=h["y"], mode="lines", name="Thực bán", line=dict(color=INK2, width=2)))
        g1.markdown("**Lịch sử bán theo tuần**")
        g1.plotly_chart(style(fig, 320, ytitle="Đơn vị / tuần"), width="stretch")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hz, y=[row[f"h{k}_q90"] for k in hz], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hz, y=[row[f"h{k}_q10"] for k in hz], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(42,120,214,0.18)", name="Dải P10–P90"))
        fig.add_trace(go.Scatter(x=hz, y=[row[f"h{k}_q50"] for k in hz], mode="lines+markers", name="P50", line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=hz, y=[row["onhand"]] * len(hz), mode="lines", name="Tồn hiện tại", line=dict(color=ORANGE, width=2)))
        g2.markdown("**Phễu bất định: nhu cầu cộng dồn so với tồn kho**")
        g2.plotly_chart(style(fig, 320, ytitle="Đơn vị cộng dồn", xtitle="Số tuần tới").update_xaxes(tickvals=hz), width="stretch")
        st.caption("Khi dải dự báo vượt lên trên đường tồn hiện tại, mã hàng có nguy cơ hết hàng trong khoảng thời gian tương ứng; "
                   "khi đường tồn nằm cao hơn nhiều so với P90 của 8 tuần, phần chênh lệch là hàng dư.")
    with r:
        tb = pd.DataFrame({"Cửa sổ": [f"{k} tuần" for k in [1, 2, 3, 4, 8]]})
        for c in ["q10", "q50", "q80", "q90", "q95"]:
            tb["P" + c[1:]] = [f"{row[f'h{k}_{c}']:.1f}" for k in [1, 2, 3, 4, 8]]
        st.markdown("**Nhu cầu cộng dồn dự báo (đơn vị)**")
        st.table(tb.set_index("Cửa sổ"))
        st.markdown(f"""
- Giá bán thực thu: **{row['price']:.2f}** · giá vốn: **{row['cost']:.2f}** · tỷ lệ thu hồi khi xả: **{row['rho']:.2f}**
- Chi phí thiếu hàng C<sub>u</sub> = **{row['cu']:.2f}** · chi phí thừa hàng C<sub>o</sub> = **{row['co']:.2f}**
- Tỷ lệ tới hạn q* = **{row['q_star']:.3f}** → mức tồn mục tiêu cho {P.protect()} tuần: **{row['target_stock']:.0f}**
- Tồn hiện tại: **{row['onhand']:.0f}** ({row['cover_weeks'] if np.isfinite(row['cover_weeks']) else 0:.0f} tuần bán) · đề xuất: **{row['action']}**
""", unsafe_allow_html=True)
    st.subheader("Độ chính xác trên giai đoạn kiểm định (13 gốc dự báo, tuần 30–45)")
    ev = D["eval"][["model", "H", "wQL", "do_phu_80", "WAPE_P50", "thien_kien_P50"]].copy()
    ev.columns = ["Mô hình", "Cửa sổ (tuần)", "wQL", "Độ phủ P10–P90", "WAPE của P50", "Thiên kiến P50"]
    ev = ev[ev["Cửa sổ (tuần)"].isin([1, 4, 8])].copy()
    for col in ev.columns[2:]:
        ev[col] = ev[col].map("{:.3f}".format)
    st.table(ev.astype({"Cửa sổ (tuần)": int}).set_index("Mô hình"))

# ------------------------------------------------------------------ 3. NHẬP HÀNG
with tabs[2]:
    st.subheader(f"Đề xuất nhập hàng (bảo vệ {P.protect()} tuần = thời gian giao hàng + chu kỳ rà soát)")
    o = dec[dec["order_qty"] > 0].copy()
    c = st.columns(3)
    c[0].metric("Số mã hàng", f"{len(o):,}"); c[1].metric("Số lượng", f"{o['order_qty'].sum():,.0f}"); c[2].metric("Giá trị (giá vốn)", money(o["order_value"].sum()))
    f1, f2 = st.columns(2)
    sup = f1.multiselect("Nhà cung cấp", sorted(o["supplier"].unique()))
    cat = f2.multiselect("Nhóm hàng", sorted(o["category"].unique()))
    if sup: o = o[o["supplier"].isin(sup)]
    if cat: o = o[o["category"].isin(cat)]
    show = o.sort_values("order_value", ascending=False)[["sku", "category", "supplier", "onhand", "sales_4w", "p50_4w", "q_star",
                                                         "safety_stock", "target_stock", "order_qty", "order_value"]]
    show.columns = ["Mã hàng", "Nhóm hàng", "Nhà cung cấp", "Tồn", "Bán 4 tuần", "P50 4 tuần tới", "q*", "Tồn an toàn", "Tồn mục tiêu", "SL nhập", "Giá trị nhập"]
    st.dataframe(show.round(2), hide_index=True, width="stretch", height=420)
    st.download_button("Tải danh sách nhập hàng (CSV)", show.to_csv(index=False).encode("utf-8-sig"), "de_xuat_nhap_hang.csv")
    by_sup = o.groupby("supplier")["order_value"].sum().sort_values().tail(10)
    fig = go.Figure(go.Bar(y=by_sup.index, x=by_sup.values, orientation="h", marker_color=BLUE))
    st.plotly_chart(style(fig, 300, xtitle="Giá trị cần nhập theo nhà cung cấp").update_layout(hovermode="y unified", bargap=0.4), width="stretch")

# ------------------------------------------------------------------ 4. HÀNG DƯ VÀ XẢ HÀNG
with tabs[3]:
    st.subheader("Hàng dư, vốn chôn và đề xuất giảm giá")
    c = st.columns(4)
    c[0].metric("Hàng dư (đơn vị)", f"{S['excess_units']:,.0f}"); c[1].metric("Vốn chôn trong hàng dư", money(S["excess_value"]))
    c[2].metric("Chi phí vốn / năm", money(S["holding_cost_year"])); c[3].metric("Lỗ dự kiến nếu xả hết", money(S["markdown_loss"]))
    order = ["Giảm giá cấp 1", "Giảm giá cấp 2", "Xả hàng"]
    av = dec[dec["action"].isin(order)].groupby("action").agg(n=("sku", "size"), value=("stock_value", "sum"), excess=("excess_value", "sum")).reindex(order).fillna(0)
    l, r = st.columns([2, 3])
    with l:
        fig = go.Figure(go.Bar(x=order, y=av["excess"], marker_color=[BLUE, BLUE, BLUE], width=0.45,
                               text=[f"{int(n)} mã" for n in av["n"]], textposition="outside"))
        st.plotly_chart(style(fig, 330, ytitle="Vốn dư theo mức đề xuất").update_layout(hovermode="x"), width="stretch")
    with r:
        m = dec[dec["action"].isin(order)].sort_values("excess_value", ascending=False)[
            ["sku", "category", "status", "age", "onhand", "sales_8w", "cover_weeks", "excess_qty", "excess_value", "action"]].copy()
        m["cover_weeks"] = m["cover_weeks"].replace(np.inf, np.nan)
        m.columns = ["Mã hàng", "Nhóm hàng", "Trạng thái hiện tại", "Tuổi (tuần)", "Tồn", "Bán 8 tuần", "Số tuần tồn", "SL dư", "Vốn dư", "Đề xuất"]
        st.dataframe(m.round(1), hide_index=True, width="stretch", height=330)
        st.download_button("Tải danh sách giảm giá (CSV)", m.to_csv(index=False).encode("utf-8-sig"), "de_xuat_giam_gia.csv")
    st.caption("Số tuần tồn = tồn hiện tại / tốc độ bán tuần (mức cao hơn giữa dự báo P50 của 8 tuần tới và thực bán 8 tuần gần nhất). Hàng dư = phần tồn vượt nhu cầu "
               f"{P.target_cover:.0f} tuần ở mức phục vụ q*.")

# ------------------------------------------------------------------ 5. ĐIỀU CHUYỂN
with tabs[4]:
    st.subheader("Đề xuất điều chuyển giữa các cửa hàng")
    c = st.columns(4)
    c[0].metric("Dòng điều chuyển", f"{S['transfer_moves']:,}"); c[1].metric("Số lượng", f"{S['transfer_units']:,.0f}")
    c[2].metric("Số cặp cửa hàng", f"{S.get('transfer_lanes', 0):,}"); c[3].metric("Biên lợi nhuận có thể giữ lại", money(S["transfer_margin"]))
    if len(tr):
        lanes = tr.groupby(["from_store", "to_store"]).agg(lines=("sku", "size"), qty=("qty", "sum"), margin=("margin_at_stake", "sum")).reset_index().sort_values("margin", ascending=False)
        lanes.columns = ["Từ cửa hàng", "Đến cửa hàng", "Số mã hàng", "Số lượng", "Biên lợi nhuận"]
        l, r = st.columns(2)
        l.markdown("**Gộp theo cặp cửa hàng (mỗi cặp một chuyến hàng)**"); l.dataframe(lanes.round(0), hide_index=True, width="stretch", height=380)
        t2 = tr.merge(dec[["sku", "category"]], on="sku")[["sku", "category", "from_store", "to_store", "qty", "value_cost", "margin_at_stake"]]
        t2.columns = ["Mã hàng", "Nhóm hàng", "Từ", "Đến", "SL", "Giá vốn", "Biên lợi nhuận"]
        r.markdown("**Chi tiết theo mã hàng**"); r.dataframe(t2.round(1), hide_index=True, width="stretch", height=380)
        st.download_button("Tải lệnh điều chuyển (CSV)", t2.to_csv(index=False).encode("utf-8-sig"), "de_xuat_dieu_chuyen.csv")
    st.caption("Cửa hàng cho: đang giữ hàng nhưng không bán được mã đó trong 8 tuần, hoặc tồn vượt 2 lần nhu cầu 8 tuần. "
               "Cửa hàng nhận: bán đều mã đó nhưng tồn thấp hơn nhu cầu của chính nó.")

# ------------------------------------------------------------------ 6. SO SÁNH KỊCH BẢN
with tabs[5]:
    st.subheader("Kịch bản đang chọn so với kịch bản cơ sở")
    rows = [("Mức phục vụ tối ưu bình quân q*", S0["q_star_mean"], S["q_star_mean"], "{:.3f}"),
            ("Số mã hàng cần nhập", S0["order_skus"], S["order_skus"], "{:,.0f}"),
            ("Giá trị cần nhập", S0["order_value"], S["order_value"], "{:,.0f}"),
            ("Vốn nằm trong hàng dư", S0["excess_value"], S["excess_value"], "{:,.0f}"),
            ("Chi phí vốn của hàng dư / năm", S0["holding_cost_year"], S["holding_cost_year"], "{:,.0f}"),
            ("Lỗ dự kiến nếu xả hàng dư", S0["markdown_loss"], S["markdown_loss"], "{:,.0f}"),
            ("Số dòng điều chuyển", S0["transfer_moves"], S["transfer_moves"], "{:,.0f}")]
    cmp_ = pd.DataFrame([{"Chỉ số": n, "Cơ sở": f.format(a), "Đang chọn": f.format(b),
                          "Thay đổi": (f"{(b - a) / a:+.1%}" if a else "–")} for n, a, b, f in rows])
    st.table(cmp_.set_index("Chỉ số"))
    st.caption("Kịch bản cơ sở: giao hàng 3 tuần, rà soát hằng tuần, chi phí vốn 20%/năm, hàng dư nằm kho 20 tuần, "
               "tỷ lệ thu hồi theo dữ liệu, tồn mục tiêu 12 tuần.")
    st.subheader("Phân bố tỷ lệ tới hạn q* theo ngành hàng")
    fig = go.Figure()
    for dv, col in DIV_COLOR.items():
        fig.add_trace(go.Histogram(x=dec.loc[dec["division"] == dv, "q_star"], name=dv, marker_color=col, opacity=0.75,
                                   xbins=dict(start=0, end=1, size=0.025)))
    fig.update_layout(barmode="overlay")
    st.plotly_chart(style(fig, 320, ytitle="Số mã hàng", xtitle="q*").update_layout(hovermode="x"), width="stretch")

# ------------------------------------------------------------------ 7. BẢN TIN TỰ ĐỘNG
with tabs[6]:
    st.subheader("Bản tin khuyến nghị sinh tự động")
    cad = st.radio("Nhịp báo cáo", ["ngày", "tuần", "tháng"], index=1, horizontal=True)
    txt = weekly_digest(dec, tr, P, cad)
    st.code(txt, language=None)
    st.download_button("Tải bản tin (TXT)", txt.encode("utf-8"), f"ban_tin_{cad}.txt")
    st.caption("Bản tin được sinh bằng luật từ kết quả mô hình. Trong phương án triển khai, một mô hình ngôn ngữ có thể viết lại "
               "bản tin này thành văn bản điều hành, nhưng mọi đề xuất vẫn phải qua bước duyệt của người phụ trách.")
