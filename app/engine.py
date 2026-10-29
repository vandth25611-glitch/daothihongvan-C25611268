# -*- coding: utf-8 -*-
"""
BỘ MÁY RA QUYẾT ĐỊNH TỒN KHO
Dùng chung cho script phân tích (05_policy.py) và ứng dụng Streamlit (app.py), nhờ đó các con số
trong tiểu luận và trên dashboard luôn khớp nhau.

Chuỗi xử lý:  tham số tài chính -> Cu, Co -> tỷ lệ tới hạn q* -> nội suy phân vị dự báo
              -> mức tồn mục tiêu -> (1) nhập hàng, (2) xả hàng, (3) điều chuyển.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

QS = np.array([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
QCOLS = ["q10", "q30", "q50", "q70", "q80", "q90", "q95"]


@dataclass
class Params:
    lead_time: int = 3            # thời gian giao hàng của nhà cung cấp (tuần) - GIẢ ĐỊNH
    review: int = 1               # chu kỳ rà soát đặt hàng (tuần)
    holding_rate: float = 0.20    # chi phí vốn lưu kho (%/năm trên giá vốn) - GIẢ ĐỊNH
    hold_weeks: float = 20.0      # số tuần một đơn vị dư nằm kho trước khi được xả
    goodwill: float = 0.0         # tổn thất uy tín khi thiếu hàng (% giá bán)
    rho_scale: float = 1.0        # hệ số điều chỉnh tỷ lệ thu hồi khi xả (1.0 = theo dữ liệu)
    rho_override: float = -1.0    # > 0: dùng một tỷ lệ thu hồi chung cho mọi nhóm hàng
    service_override: float = -1.0  # > 0: bỏ qua q*, dùng một mức phục vụ chung
    target_cover: float = 12.0    # số tuần tồn mục tiêu để xác định hàng dư
    md1_cover: float = 26.0       # ngưỡng số tuần tồn -> đề xuất giảm giá cấp 1
    md2_cover: float = 39.0       # -> giảm giá cấp 2
    clear_cover: float = 52.0     # -> xả hàng
    min_age: int = 8              # mã hàng mới dưới ngần này tuần thì chưa đề xuất giảm giá
    tr_min_sales: int = 2         # cửa hàng nhận phải bán được ít nhất ngần này đơn vị trong 8 tuần

    def protect(self):            # số tuần cần bảo vệ = thời gian giao hàng + chu kỳ rà soát
        return int(min(max(self.lead_time + self.review, 1), 8))


def critical_ratio(price, cost, rho, p: Params):
    """Tỷ lệ tới hạn Newsvendor cho từng mã hàng.
    Cu = (giá bán thực thu - giá vốn) + tổn thất uy tín
    Co = giá vốn x (1 - tỷ lệ thu hồi khi xả) + chi phí vốn lưu kho trong thời gian chờ xả"""
    rho_eff = np.where(p.rho_override > 0, p.rho_override, np.minimum(rho * p.rho_scale, 1.0))
    cu = np.maximum(price - cost, 0.0) + p.goodwill * price
    co = cost * (1.0 - rho_eff) + cost * p.holding_rate * p.hold_weeks / 52.0
    q = cu / np.maximum(cu + co, 1e-9)
    if p.service_override > 0:
        q = np.full_like(q, p.service_override, dtype=float)
    return cu, co, q


def quantile_at(qmat, q):
    """Nội suy tuyến tính trên đường phân vị dự báo (7 mốc P10..P95) tại mức q của từng mã hàng.
    q ngoài khoảng [0.10, 0.95] được kẹp về biên."""
    q = np.clip(np.asarray(q, dtype=float), QS[0], QS[-1])
    j = np.clip(np.searchsorted(QS, q, side="right") - 1, 0, len(QS) - 2)
    r = np.arange(len(q))
    lo, hi = qmat[r, j], qmat[r, j + 1]
    w = (q - QS[j]) / (QS[j + 1] - QS[j])
    return lo + w * (hi - lo)


def horizon_quantile(fc_wide, H, q):
    """Phân vị q của nhu cầu cộng dồn H tuần. Bảng dự báo có các mốc H = 1, 2, 3, 4, 8;
    các H ở giữa (5, 6, 7) được nội suy tuyến tính giữa mốc 4 và mốc 8."""
    def get(h):
        return quantile_at(fc_wide[[f"h{h}_{c}" for c in QCOLS]].values, q)
    if H in (1, 2, 3, 4, 8):
        return get(H)
    a, b = get(4), get(8)
    return a + (b - a) * (H - 4) / 4.0


STATUS_TIER = {"Full Price": 0, "Promo": 0, "Markdown Tier 1": 1, "Markdown Tier 2": 2, "Clearance": 3}
ACTIONS = ["Giữ nguyên", "Giảm giá cấp 1", "Giảm giá cấp 2", "Xả hàng"]


def decide(sku: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Tính toàn bộ quyết định ở cấp mã hàng.
    sku: mỗi dòng một mã hàng, gồm giá, giá vốn, rho, tồn hiện tại, tuổi, trạng thái,
         dự báo thường (h{H}_qXX) và dự báo 'nếu đủ hàng' (w{H}_qXX)."""
    d = sku.copy()
    cu, co, q = critical_ratio(d["price"].values, d["cost"].values, d["rho"].values, p)
    d["cu"], d["co"], d["q_star"] = cu, co, q
    H = p.protect()
    wi = d[[c for c in d.columns if c.startswith("w")]].rename(columns=lambda c: "h" + c[1:])
    d["p50_4w"] = horizon_quantile(d, 4, np.full(len(d), 0.5))
    d["p50_8w"] = horizon_quantile(d, 8, np.full(len(d), 0.5))
    # Tốc độ bán tuần: lấy mức cao hơn giữa dự báo P50 và thực bán 8 tuần gần nhất. Với nhu cầu thưa, trung vị
    # thấp hơn kỳ vọng, nên chỉ dùng P50 sẽ thổi phồng số tuần tồn và đề xuất giảm giá quá tay.
    recent = d["sales_8w"] if "sales_8w" in d.columns else 0.0
    best = np.maximum(d["p50_8w"], recent)
    d["weekly_rate"] = np.maximum(best / 8.0, 1e-6)
    d["cover_weeks"] = np.where(best > 0, d["onhand"] / d["weekly_rate"], np.inf)

    # (1) NHẬP HÀNG: mức tồn mục tiêu S = phân vị q* của nhu cầu trong (L + R) tuần, dùng dự báo 'nếu đủ hàng'
    d["target_stock"] = np.ceil(horizon_quantile(wi, H, q) - 1e-9)
    d["order_qty"] = np.maximum(d["target_stock"] - d["onhand"], 0)
    sellable = d["status"].isin(["Full Price", "Promo"])          # không nhập thêm mã đang giảm giá/xả
    d.loc[~sellable, "order_qty"] = 0
    d["order_value"] = d["order_qty"] * d["cost"]
    d["safety_stock"] = np.maximum(d["target_stock"] - horizon_quantile(wi, H, np.full(len(d), 0.5)), 0)

    # (2) HÀNG DƯ VÀ VỐN CHÔN: tồn vượt mức cần cho target_cover tuần ở mức phục vụ q*
    need_cover = horizon_quantile(d, 8, q) * p.target_cover / 8.0
    d["excess_qty"] = np.maximum(d["onhand"] - np.ceil(need_cover), 0)
    d["stock_value"] = d["onhand"] * d["cost"]
    d["excess_value"] = d["excess_qty"] * d["cost"]
    d["holding_cost_year"] = d["excess_value"] * p.holding_rate
    rho_eff = np.where(p.rho_override > 0, p.rho_override, np.minimum(d["rho"] * p.rho_scale, 1.0))
    d["markdown_loss"] = d["excess_value"] * (1 - rho_eff)

    # (3) ĐỀ XUẤT GIẢM GIÁ theo số tuần tồn
    cov = d["cover_weeks"].replace(np.inf, 999)
    old = d["age"] >= p.min_age
    ok = old & (d["excess_qty"] > 0)                               # chỉ xét mã đã bán đủ lâu và thật sự có hàng dư
    tier = np.zeros(len(d), dtype=int)
    tier[(ok & (cov > p.md1_cover)).values] = 1
    tier[(ok & (cov > p.md2_cover)).values] = 2
    tier[(ok & (cov > p.clear_cover)).values] = 3
    current = d["status"].map(STATUS_TIER).fillna(0).astype(int).values
    tier[tier <= current] = 0                                      # mã đã ở mức giảm giá đó (hoặc sâu hơn) thì giữ nguyên
    d["action"] = np.array(ACTIONS)[tier]
    d.loc[d["order_qty"] > 0, "action"] = "Nhập hàng"
    return d


def transfers(dec: pd.DataFrame, store_stock: pd.DataFrame, p: Params, max_moves=None) -> pd.DataFrame:
    """Đề xuất điều chuyển giữa các cửa hàng cho từng mã hàng.
    - Cửa hàng nhận: bán được mã hàng đều trong 8 tuần gần nhất (>= tr_min_sales đơn vị) nhưng tồn thấp hơn
      nhu cầu (L + R) tuần của chính nó.
    - Cửa hàng cho: đang giữ hàng nhưng không bán được trong 8 tuần, hoặc tồn vượt 2 lần nhu cầu 8 tuần.
    store_stock: store, sku, onhand, sales_8w."""
    H = p.protect()
    s = store_stock.merge(dec[["sku", "target_stock", "p50_8w", "cost", "price", "status"]], on="sku", how="inner")
    tot = s.groupby("sku")["sales_8w"].transform("sum")
    s["share"] = np.where(tot > 0, s["sales_8w"] / tot.replace(0, np.nan), 0.0)
    s["need"] = np.round(s["share"] * s["target_stock"])                    # nhu cầu (L+R) tuần của cửa hàng
    s["need_8w"] = np.ceil(s["share"] * s["p50_8w"] - 1e-9)
    s["deficit"] = np.where(s["sales_8w"] >= p.tr_min_sales, np.maximum(s["need"] - s["onhand"], 0), 0)
    s["surplus"] = np.where(s["sales_8w"] == 0, s["onhand"], np.maximum(s["onhand"] - 2 * s["need_8w"], 0))
    s.loc[s["deficit"] > 0, "surplus"] = 0
    moves = []
    for sku, g in s[(s["deficit"] > 0) | (s["surplus"] > 0)].groupby("sku"):
        rec = g[g["deficit"] > 0].sort_values("deficit", ascending=False)[["store", "deficit"]].values.tolist()
        don = g[g["surplus"] > 0].sort_values("surplus", ascending=False)[["store", "surplus"]].values.tolist()
        if not rec or not don:
            continue
        cost = g["cost"].iloc[0]; price = g["price"].iloc[0]
        i = j = 0
        while i < len(rec) and j < len(don):
            qty = min(rec[i][1], don[j][1])
            moves.append((sku, don[j][0], rec[i][0], qty, qty * cost, qty * (price - cost)))
            rec[i][1] -= qty; don[j][1] -= qty
            if rec[i][1] <= 0: i += 1
            if don[j][1] <= 0: j += 1
    out = pd.DataFrame(moves, columns=["sku", "from_store", "to_store", "qty", "value_cost", "margin_at_stake"])
    out = out.sort_values("margin_at_stake", ascending=False).reset_index(drop=True)
    return out if max_moves is None else out.head(max_moves)


def summary(dec: pd.DataFrame, tr: pd.DataFrame, p: Params) -> dict:
    """Các chỉ số tổng hợp cho màn hình tổng quan và bản tin."""
    w = dec["stock_value"].sum()
    return {
        "n_sku": int(len(dec)), "stock_units": float(dec["onhand"].sum()), "stock_value": float(w),
        "excess_units": float(dec["excess_qty"].sum()), "excess_value": float(dec["excess_value"].sum()),
        "excess_share": float(dec["excess_value"].sum() / max(w, 1e-9)),
        "holding_cost_year": float(dec["holding_cost_year"].sum()),
        "markdown_loss": float(dec["markdown_loss"].sum()),
        "order_skus": int((dec["order_qty"] > 0).sum()), "order_units": float(dec["order_qty"].sum()),
        "order_value": float(dec["order_value"].sum()),
        "q_star_mean": float(np.average(dec["q_star"], weights=np.maximum(dec["p50_8w"], 1e-6))),
        "actions": dec["action"].value_counts().to_dict(),
        "transfer_moves": int(len(tr)), "transfer_units": float(tr["qty"].sum()) if len(tr) else 0.0,
        "transfer_margin": float(tr["margin_at_stake"].sum()) if len(tr) else 0.0,
        "transfer_lanes": int(tr.groupby(["from_store", "to_store"]).ngroups) if len(tr) else 0,
        "params": asdict(p)}


def weekly_digest(dec: pd.DataFrame, tr: pd.DataFrame, p: Params, cadence="tuần", top=5) -> str:
    """Bản tin khuyến nghị sinh tự động bằng luật (không cần mô hình ngôn ngữ)."""
    s = summary(dec, tr, p)
    f = lambda v: f"{v:,.0f}"
    lines = [f"BẢN TIN TỒN KHO ({cadence.upper()})",
             f"- Tồn kho hiện tại: {f(s['stock_units'])} đơn vị, giá trị {f(s['stock_value'])} theo giá vốn.",
             f"- Vốn nằm trong hàng dư (vượt {p.target_cover:.0f} tuần bán): {f(s['excess_value'])} "
             f"({s['excess_share']:.0%} giá trị tồn); chi phí vốn ước tính {f(s['holding_cost_year'])}/năm."]
    if cadence in ("ngày", "tuần"):
        o = dec[dec["order_qty"] > 0].sort_values("order_value", ascending=False)
        lines.append(f"- Cần nhập: {s['order_skus']} mã hàng, {f(s['order_units'])} đơn vị, {f(s['order_value'])} giá vốn.")
        for _, r in o.head(top).iterrows():
            lines.append(f"    • {r['sku']} ({r['category']}): tồn {r['onhand']:.0f}, mục tiêu {r['target_stock']:.0f} "
                         f"(q* = {r['q_star']:.2f}) → nhập {r['order_qty']:.0f}")
    if cadence in ("tuần",):
        lines.append(f"- Điều chuyển: {s['transfer_moves']:,} lệnh, {f(s['transfer_units'])} đơn vị, "
                     f"biên lợi nhuận có thể giữ lại {f(s['transfer_margin'])}.")
    if cadence in ("tuần", "tháng"):
        a = s["actions"]
        lines.append(f"- Đề xuất giảm giá: cấp 1: {a.get('Giảm giá cấp 1', 0)} mã; cấp 2: {a.get('Giảm giá cấp 2', 0)} mã; "
                     f"xả hàng: {a.get('Xả hàng', 0)} mã. Lỗ dự kiến nếu xả toàn bộ hàng dư: {f(s['markdown_loss'])}.")
    if cadence == "tháng":
        c = dec.groupby("category").agg(v=("stock_value", "sum"), e=("excess_value", "sum")).sort_values("e", ascending=False)
        lines.append("- Nhóm hàng có vốn dư lớn nhất:")
        for k, r in c.head(top).iterrows():
            lines.append(f"    • {k}: dư {f(r['e'])} / tồn {f(r['v'])} ({r['e']/max(r['v'],1e-9):.0%})")
        lines.append(f"- Mức phục vụ tối ưu bình quân (q*): {s['q_star_mean']:.2f}. Cần rà lại các tham số giả định: "
                     f"thời gian giao hàng {p.lead_time} tuần, lãi suất vốn {p.holding_rate:.0%}/năm.")
    lines.append("Mọi đề xuất cần người phụ trách duyệt trước khi thực hiện.")
    return "\n".join(lines)
