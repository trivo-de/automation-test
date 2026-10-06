"""Giao diện recommendation từ research signal + hành vi khách hàng."""

import json

import gradio as gr
import lightgbm as lgb
import numpy as np
import pandas as pd

import pipeline
from config import GRADIO_PORT, MARKET_WEIGHT, OUT_DIR, TOPK, connect, log


def load_model(name):
    model = lgb.Booster(model_file=f"{OUT_DIR}/model_{name}.txt")
    with open(f"{OUT_DIR}/meta_{name}.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def score(df, model, meta):
    x = df.copy()
    for c in meta["cats"]:
        x[c] = pd.Categorical(
            x[c].astype("string").fillna("UNKNOWN"),
            categories=meta["categories"][c],
        )
    for c in meta["features"]:
        if c not in meta["cats"]:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype("float32")
    return model.predict(x[meta["features"]])


def minmax(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).any():
        return np.zeros(len(values))
    lo, hi = np.nanmin(values), np.nanmax(values)
    if hi - lo <= 1e-12:
        return np.zeros(len(values))
    return np.nan_to_num((values - lo) / (hi - lo))


def research_action(value):
    # BUY chỉ là tín hiệu mới trong ngày; BUY cũ còn mở được hiển thị là HOLD.
    return {
        "SELL": "BÁN",
        "BUY": "CÓ THỂ MUA",
        "HOLD": "GIỮ",
    }.get(value, "CHƯA CÓ TÍN HIỆU")


def probability(raw):
    """Đầu ra của model là xác suất thật (mục tiêu binary, không cân bằng lại lớp)."""
    return (100 * pd.Series(np.asarray(raw, dtype=float))).round(1).astype(str) + "%"


def fmt(value, digits=3):
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def signed_pct(value):
    return "—" if value is None or pd.isna(value) else f"{100 * value:+.2f}%"


BRANCHES = (("buy", "MUA"), ("portfolio", "Danh mục"))
SCORING = {
    "baseline": "Chỉ tín hiệu PHS",
    "behavior": "Chỉ model hành vi",
    "final": "Điểm cuối (đang dùng)",
}
AUDIT_ROWS = {  # khoá -> (nhãn, có phải tỷ lệ phần trăm không)
    "basket_excess": ("Cả rổ PHS BUY mới so với VNINDEX", True),
    "top_minus_basket": ("Mã MarketScore cao nhất so với trung bình rổ", True),
    "rank_ic": ("Tương quan hạng giữa MarketScore và lợi suất", False),
}


def ranking_table(summary):
    """HitRate@1 và các thước đo xếp hạng của từng cách chấm điểm, mỗi nhánh."""
    lines = [
        "| Nhánh | Cách chấm điểm | HitRate@1 | NDCG@3 | MRR | "
        "Precision@1 (mọi nhóm) | Lợi suất vượt trội của mã đầu |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    notes = []
    for key, title in BRANCHES:
        result = summary.get(key)
        if not result:
            continue
        lines.append(
            f"| {title} | Chọn ngẫu nhiên | {fmt(result.get('random_hit@1'))} "
            "| — | — | — | — |"
        )
        for variant, label in SCORING.items():
            m = result.get(variant, {})
            lines.append(
                f"| {title} | {label} | {fmt(m.get('hit@1'))} | {fmt(m.get('ndcg@3'))} "
                f"| {fmt(m.get('mrr'))} | {fmt(m.get('precision@1'))} "
                f"| {signed_pct(m.get('excess@1'))} |"
            )
        notes.append(
            f"{title}: AUC hành vi {result['auc']} · {result['n_folds']} quý test · "
            f"{result['n_groups']:,} nhóm xếp hạng được trên "
            f"{result['n_groups_all']:,} nhóm · model cuối {result['rounds']} vòng."
        )
    return "\n".join(lines) + "\n\n" + "\n".join(notes)


def weights_table(weights):
    """Nhánh MUA ở từng mức MARKET_WEIGHT, trên cùng một bộ dự đoán ngoài mẫu."""
    lines = [
        "### Nhánh MUA ở các mức MARKET_WEIGHT khác nhau\n",
        "Cùng một bộ dự đoán ngoài mẫu, chỉ đổi tỷ lệ trộn. 0 = chỉ hành vi, "
        "1 = chỉ MarketScore.\n",
        "| MARKET_WEIGHT | HitRate@1 | NDCG@3 | Precision@1 | "
        "Lợi suất vượt trội của mã đầu |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in weights:
        current = abs(row["market_weight"] - MARKET_WEIGHT) < 1e-9
        lines.append(
            f"| {row['market_weight']:.1f}{' ← đang dùng' if current else ''} "
            f"| {fmt(row.get('hit@1'))} | {fmt(row.get('ndcg@3'))} "
            f"| {fmt(row.get('precision@1'))} | {signed_pct(row.get('excess@1'))} |"
        )
    return "\n".join(lines)


def audit_table(audit):
    """MarketScore có dự báo lợi suất không — đo trên rổ PHS BUY mới."""
    lines = [
        "### MarketScore có dự báo lợi suất không?\n",
        f"Lợi suất vượt VNINDEX sau {audit['horizon']} phiên, mỗi mốc là một quan sát. "
        "Sai số chuẩn đã nới theo số cửa sổ không chồng nhau. |t| dưới 2 nghĩa là "
        "chưa đủ bằng chứng.\n",
        "| Phép đo | Số mốc | Trung bình | Sai số chuẩn | t | Tỷ lệ mốc dương |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, (label, as_pct) in AUDIT_ROWS.items():
        m = audit.get(key) or {}
        if "mean" not in m:
            continue
        mean = signed_pct(m["mean"]) if as_pct else f"{m['mean']:+.3f}"
        se = f"{100 * m['se']:.2f}%" if as_pct else f"{m['se']:.3f}"
        lines.append(
            f"| {label} | {m['n']} | {mean} | {se} | {fmt(m.get('t_stat'), 2)} "
            f"| {m['share_positive']:.0%} |"
        )
    return "\n".join(lines)


def performance_markdown(summary):
    freq = {"day": "ngày", "week": "tuần", "month": "tháng"}.get(
        summary.get("decision_freq"), "?"
    )
    sections = [
        "### Hiệu quả walk-forward theo quý (dự đoán ngoài mẫu)",
        f"Mốc quyết định theo **{freq}**. Nhánh BUY chỉ xét mã PHS vừa mở khuyến "
        f"nghị `BUY` trong đúng ngày đó. Điểm cuối = **{MARKET_WEIGHT:.0%} MarketScore + "
        f"{1 - MARKET_WEIGHT:.0%} hành vi**, trong đó MarketScore = dư địa tăng từ "
        "giá hiện tại tới giá mục tiêu.",
        "**HitRate@1** = tỷ lệ nhóm (khách, mốc) mà mã xếp đầu đúng là mã khách "
        "mua/bán. Chỉ tính trên nhóm *có thể xếp sai* (có cả mã đúng lẫn mã sai); "
        "nhóm chỉ có 1 ứng viên bị loại vì luôn đạt 1,0. Số liệu gộp mọi nhóm của "
        "mọi quý.",
        ranking_table(summary),
    ]
    weights = (summary.get("buy") or {}).get("weights")
    if weights:
        sections.append(weights_table(weights))
    if summary.get("market_score_audit"):
        sections.append(audit_table(summary["market_score_audit"]))
    sections.append(
        "Các thước đo trên đo mức phù hợp hành vi. Tín hiệu MUA / chốt lời / cắt lỗ "
        "là đầu vào từ khuyến nghị PHS, không phải do model hành vi tự tạo."
    )
    return "\n\n".join(sections)


HORIZON_LABEL = {
    "SHORT": "NGẮN HẠN",
    "LEAN_SHORT": "THIÊN NGẮN HẠN",
    "MEDIUM": "TRUNG HẠN (1-3 tháng)",
    "LEAN_LONG": "THIÊN DÀI HẠN",
    "LONG": "DÀI HẠN",
    "UNRATED": "CHƯA ĐÁNH GIÁ ĐƯỢC",
}
CALL_LABEL = {
    "BUY": "PHS mới khuyến nghị MUA",
    "HOLD": "PHS chưa có tín hiệu mới",
    "TAKE_PROFIT": "PHS đã chốt lời",
    "CUT_LOSS": "PHS đã cắt lỗ",
}


def pct(value):
    return "—" if value is None or pd.isna(value) else f"{100 * value:.0f}%"


def build():
    con = connect(read_only=True)
    buy_model, buy_meta = load_model("buy")
    portfolio_model, portfolio_meta = load_model("portfolio")
    dates = [
        str(r[0])
        for r in con.execute("SELECT t FROM decision_dates ORDER BY t").fetchall()
    ]
    customers = [
        r[0]
        for r in con.execute(
            "SELECT customer_id FROM prof ORDER BY customer_id"
        ).fetchall()
    ]
    names = (
        con.execute("SELECT stock_code, company_name, icb_name FROM sec")
        .df()
        .set_index("stock_code")
    )
    with open(f"{OUT_DIR}/summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    def decorate(df):
        return df.join(names, on="stock_code")

    def recommend(customer_id, t):
        info = con.execute(
            "SELECT customer_type, open_date FROM prof WHERE customer_id = ?",
            [customer_id],
        ).fetchone()
        # Kỳ hạn là nhãn point-in-time: chỉ dựa trên các vòng nắm giữ trước t.
        hz = con.execute(
            "SELECT investment_horizon, n_evidence, short_ratio, long_ratio, "
            "median_hold_days, win_rate, mid_ratio "
            "FROM cust_horizon WHERE customer_id = ? AND t = ?",
            [customer_id, t],
        ).fetchone()
        header = f"**{customer_id}** — {info[0]} · mở TK {info[1]}  \n"
        if hz is None:
            header += "Kỳ hạn đầu tư: **CHƯA ĐÁNH GIÁ ĐƯỢC** (chưa có giao dịch trước mốc này)"
        else:
            hold = "—" if hz[4] is None else f"{hz[4]:.0f} ngày"
            header += (
                f"Kỳ hạn đầu tư: **{HORIZON_LABEL.get(hz[0], hz[0])}** — "
                f"{hz[1]} vòng mua-bán · ngắn hạn {pct(hz[2])} · 1-3 tháng {pct(hz[6])} · "
                f"dài hạn {pct(hz[3])} · "
                f"giữ trung vị {hold} · tỷ lệ vòng có lãi {pct(hz[5])}"
            )
        header += f"  \n*Thời điểm: {t}*"

        # BUY: hard-filter đã được thực hiện ở train_buy, mọi dòng đều là mã PHS
        # mới mở khuyến nghị MUA đúng ngày t.
        buy_rows = con.execute(
            "SELECT * FROM train_buy WHERE customer_id = ? AND t = ?",
            [customer_id, t],
        ).df()
        if buy_rows.empty:
            buy_table = pd.DataFrame(
                {"Thông báo": ["PHS không có khuyến nghị MUA mới trong ngày này"]}
            )
        else:
            raw = score(buy_rows, buy_model, buy_meta)
            buy_rows["buy_probability"] = probability(raw).to_numpy()
            # co giãn trong rổ của khách chỉ để trộn với MarketScore
            buy_rows["behavior_score"] = minmax(raw)
            buy_rows["final_score"] = (
                MARKET_WEIGHT * buy_rows["market_score_norm"].fillna(0)
                + (1 - MARKET_WEIGHT) * buy_rows["behavior_score"]
            )
            buy_rows = decorate(
                buy_rows.sort_values("final_score", ascending=False).head(TOPK)
            )
            buy_table = pd.DataFrame(
                {
                    "Mã": buy_rows.stock_code,
                    "Công ty": buy_rows.company_name,
                    "Ngành": buy_rows.icb_name,
                    "Tín hiệu": "BUY",
                    "Giá mục tiêu": buy_rows.target_price.round(0),
                    "Giá cắt lỗ": buy_rows.cut_loss_price.round(0),
                    "Dư địa tới mục tiêu (%)": buy_rows.market_score.round(2),
                    "Khả năng khách mua (model)": buy_rows.buy_probability,
                    "Điểm hành vi (0-1 trong rổ)": buy_rows.behavior_score.round(4),
                    "Điểm cuối": buy_rows.final_score.round(4),
                    "Khách tương tự vừa mua": buy_rows.n_peer_buyers.astype(int),
                    "Graph": buy_rows.graph_score.round(4),
                    "Style": buy_rows.suitability_score.round(4),
                    "Lịch sử đúng mã": buy_rows.historical_preference.round(4),
                    "Đang nắm": buy_rows.is_holding.map({1: "có", 0: ""}),
                }
            )

        # PORTFOLIO: action do research quyết định; model chỉ cho biết mức ưu tiên
        # hành vi bán để sắp xếp các vị thế cùng tín hiệu.
        portfolio_rows = con.execute(
            "SELECT * FROM train_portfolio WHERE customer_id = ? AND t = ?",
            [customer_id, t],
        ).df()
        if portfolio_rows.empty:
            portfolio_table = pd.DataFrame(
                {"Thông báo": ["Khách không có vị thế tại mốc này"]}
            )
        else:
            raw = score(portfolio_rows, portfolio_model, portfolio_meta)
            portfolio_rows["sell_behavior_score"] = raw
            portfolio_rows["sell_probability"] = probability(raw).to_numpy()
            portfolio_rows["action"] = portfolio_rows.research_recommendation.map(
                research_action
            )
            priority = {"BÁN": 0, "CÓ THỂ MUA": 1, "GIỮ": 2, "CHƯA CÓ TÍN HIỆU": 3}
            portfolio_rows["action_priority"] = portfolio_rows.action.map(priority)
            portfolio_rows = decorate(
                portfolio_rows.sort_values(
                    ["action_priority", "sell_behavior_score"], ascending=[True, False]
                )
            )
            portfolio_table = pd.DataFrame(
                {
                    "Mã": portfolio_rows.stock_code,
                    "Công ty": portfolio_rows.company_name,
                    "Tín hiệu nghiên cứu": portfolio_rows.research_call_type.map(
                        CALL_LABEL
                    ).fillna("—"),
                    "Khuyến nghị": portfolio_rows.action,
                    "Khả năng khách bán (model)": portfolio_rows.sell_probability,
                    "Khách tương tự vừa bán": portfolio_rows.n_peer_sellers.astype(int),
                    "Đã giữ (ngày)": portfolio_rows.pos_hold_days,
                    "Lãi/lỗ hiện tại": (100 * portfolio_rows.unreal_pnl_pct)
                    .round(1)
                    .astype(str)
                    + "%",
                    "Tỷ trọng": (100 * portfolio_rows.weight_in_port)
                    .round(1)
                    .astype(str)
                    + "%",
                }
            )
        return header, buy_table, portfolio_table

    perf = performance_markdown(summary)

    with gr.Blocks(title="Khuyến nghị cổ phiếu cá nhân hóa") as ui:
        gr.Markdown(
            "# Khuyến nghị từ tín hiệu nghiên cứu + hành vi khách hàng\n"
            "Chỉ xếp hạng mã PHS mới khuyến nghị **MUA** trong ngày; danh mục dùng tín hiệu "
            "**MUA / GIỮ / chốt lời / cắt lỗ** của PHS."
        )
        with gr.Tab("Khuyến nghị"):
            with gr.Row():
                customer = gr.Dropdown(
                    customers, value=customers[0], label="Khách hàng"
                )
                date = gr.Dropdown(dates, value=dates[-1], label="Thời điểm t")
                button = gr.Button("Gợi ý", variant="primary")
            heading = gr.Markdown()
            gr.Markdown(f"### TOP-{TOPK} mã BUY phù hợp nhất")
            buy_output = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown("### Danh mục hiện tại — tín hiệu BÁN / MUA / GIỮ")
            portfolio_output = gr.Dataframe(interactive=False, wrap=True)
            button.click(
                recommend,
                [customer, date],
                [heading, buy_output, portfolio_output],
            )
        with gr.Tab("Hiệu quả model"):
            gr.Markdown(perf)
    # Để scripts/verify_app.py gọi thẳng hàm gợi ý, không phải đi qua giao diện.
    ui.recommend = recommend
    return ui


if __name__ == "__main__":
    pipeline.wait_until_done()
    log("app", f"khởi động Gradio ở cổng {GRADIO_PORT}")
    build().launch(server_name="0.0.0.0", server_port=GRADIO_PORT, show_api=False)
