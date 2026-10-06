"""Bước 5 — Huấn luyện model hành vi cho BUY và SELL.

Danh sách research đã quyết định mã nào có tín hiệu BUY/SELL. Hai model ở đây chỉ
học mức phù hợp với từng khách:

* BUY: xác suất khách mua một mã trong rổ research BUY.
* PORTFOLIO: xác suất khách bán một mã đang sở hữu.

Điểm BUY cuối = MarketScore (chất lượng từ research) + BehaviorScore (model).

Cách đánh giá (walk-forward theo quý, không chia ngẫu nhiên):
  * Có khoảng đệm HORIZON phiên giữa train và test: một dòng train chỉ được dùng
    khi cửa sổ nhãn của nó đã kết thúc trước mốc test đầu tiên.
  * Số vòng boosting chọn bằng dừng sớm trên quý train cuối, không đặt tay.
  * Thước đo xếp hạng chỉ tính trên nhóm (khách, mốc) CÓ THỂ xếp sai: có ít nhất
    một mã đúng và một mã sai. Nhóm chỉ có 1 ứng viên luôn đạt 1,0 nên bị loại.
  * Số tổng hợp là gộp mọi nhóm của mọi quý, không phải trung bình các quý.
"""

import fnmatch
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import (
    ABLATE_FEATURES,
    DECISION_FREQ,
    EVAL_KS,
    GRAPH_BACKEND,
    HORIZON,
    MARKET_WEIGHT,
    MARKET_WEIGHT_GRID,
    OUT_DIR,
    RESEARCH_TIEBREAK,
    SAMPLE_PCT,
    connect,
    log,
)

CATS = [
    "customer_type",
    "investment_horizon",
    "exchange",
    "icb_code",
    "last_side",
    "research_recommendation",
]
BASE_IDS = [
    "t",
    "dn",
    "customer_id",
    "stock_code",
    "y",
    "excess_fwd",
    "label_complete",
    "q",
    "g",
    "c",
]

PARAMS = {
    # "binary" cho xác suất thật để hiển thị. Đã thử "lambdarank" (02/10/2026):
    # HitRate@1 như nhau ở cả hai nhánh nên không giữ tuỳ chọn đó.
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
    "seed": 42,
}
RESEARCH_COLUMN_PREFIXES = ("market_score", "candidate_r", "research_", "in_research")
# các thước đo ghi vào bảng theo quý và bảng MARKET_WEIGHT
REPORT_KEYS = ("hit@1", "ndcg@3", "mrr", "precision@1", "excess@1")
MAX_ROUNDS = 400
EARLY_STOP = 30
MIN_ROUNDS = 20
DEFAULT_ROUNDS = 300
MIN_TRAIN_QUARTERS = 4


def prep(df, extra_drop=()):
    """Tách feature khỏi id/nhãn và chốt thứ tự nhãn của các cột phân loại.

    load_table đã trả số thực dạng float32 và chuỗi dạng category, nên ở đây không
    phải ép kiểu lại (bảng danh mục 1,75 triệu dòng: mỗi bản sao thừa là vài trăm MB).
    """
    blocked = set(BASE_IDS) | set(extra_drop)
    feats = [c for c in df.columns if c not in blocked]
    for c in CATS:
        if c in feats:
            col = df[c].astype("category").cat.rename_categories(str)
            if col.isna().any():
                col = col.cat.set_categories(
                    list(dict.fromkeys([*col.cat.categories, "UNKNOWN"]))
                ).fillna("UNKNOWN")
            # thứ tự nhãn cố định -> mã số của nhãn không phụ thuộc thứ tự dòng
            df[c] = col.cat.reorder_categories(sorted(col.cat.categories))
    return df, feats


def normalize_within_group(g, score):
    """Đưa score về [0,1] riêng trong mỗi khách/mốc để ghép với MarketScore."""
    z = pd.Series(np.asarray(score, dtype=float))
    by = z.groupby(np.asarray(g))
    lo, hi = by.transform("min"), by.transform("max")
    den = hi - lo
    return ((z - lo) / den.where(den > 1e-12, 1.0)).fillna(0.0).to_numpy()


def ranking_metrics(g, y, excess, score, ks=EVAL_KS, cluster=None):
    """Thước đo xếp hạng theo từng nhóm (khách, mốc).

    g: mã nhóm của từng dòng; y: nhãn 0/1; excess: lợi suất vượt trội để audit.

    * precision@1, excess@1: trên MỌI nhóm — mã đứng đầu có đúng không, lời không.
    * hit@k, recall@k, ndcg@k, mrr: chỉ trên nhóm có cả mã đúng lẫn mã sai. Nhóm
      toàn đúng hoặc chỉ có 1 ứng viên thì cách xếp nào cũng đạt 1,0.
    * random_hit@1: hit@1 kỳ vọng nếu chọn ngẫu nhiên — mốc so sánh thấp nhất.
    * hit@1_se (khi có `cluster` = mã khách của từng dòng): sai số chuẩn của hit@1
      tính theo CỤM khách. Các nhóm của cùng một khách ở các tuần liền nhau gần như
      lặp lại nhau, nên coi mỗi nhóm là một quan sát độc lập sẽ cho sai số nhỏ giả tạo.
    """
    d = pd.DataFrame(
        {
            "g": np.asarray(g),
            "y": np.asarray(y, dtype=float),
            "x": np.asarray(excess, dtype=float),
            "s": np.asarray(score, dtype=float),
        }
    )
    by = d.groupby("g", sort=False)
    d["rk"] = by["s"].rank(ascending=False, method="first")
    d["n"] = by["y"].transform("size")
    d["npos"] = by["y"].transform("sum")

    top1 = d[d["rk"] == 1]
    out = {
        "n_groups_all": len(top1),
        "precision@1": float(top1["y"].mean()) if len(top1) else np.nan,
        "excess@1": float(top1["x"].mean()) if len(top1) else np.nan,
    }

    ev = d[(d["npos"] > 0) & (d["npos"] < d["n"])]
    first_pos = ev[ev["y"] == 1].groupby("g")["rk"].min()
    out["n_groups"] = len(first_pos)
    if first_pos.empty:
        names = ["random_hit@1", "mrr"]
        names += [f"{m}@{k}" for k in ks for m in ("hit", "recall", "ndcg")]
        return out | dict.fromkeys(names, np.nan)

    head = ev.groupby("g")[["npos", "n"]].first().reindex(first_pos.index)
    out["random_hit@1"] = float((head["npos"] / head["n"]).mean())
    out["mrr"] = float((1.0 / first_pos).mean())
    if cluster is not None:
        owner = pd.Series(np.asarray(cluster), index=d["g"]).groupby(level=0).first()
        out["hit@1_se"] = clustered_se(first_pos == 1, owner.reindex(first_pos.index))
    for k in ks:
        topk = ev[ev["rk"] <= k]
        hits = topk.groupby("g")["y"].sum().reindex(first_pos.index, fill_value=0)
        gain = topk["y"] / np.log2(topk["rk"] + 1)
        dcg = gain.groupby(topk["g"]).sum().reindex(first_pos.index, fill_value=0)
        ideal = np.cumsum(1.0 / np.log2(np.arange(k) + 2))
        idcg = ideal[np.minimum(head["npos"].to_numpy(), k).astype(int) - 1]
        out[f"hit@{k}"] = float((first_pos <= k).mean())
        out[f"recall@{k}"] = float((hits / head["npos"]).mean())
        out[f"ndcg@{k}"] = float((dcg / idcg).mean())
    return out


def clustered_se(hits, cluster):
    """Sai số chuẩn của một tỷ lệ khi các quan sát cùng cụm tương quan với nhau.

    Công thức ước lượng tỷ số: phần dư được cộng dồn theo cụm trước khi bình phương,
    nên một khách có 100 nhóm giống nhau chỉ được tính như một nguồn thông tin.
    """
    h = np.asarray(hits, dtype=float)
    p = h.mean()
    resid = pd.Series(h - p).groupby(np.asarray(cluster)).sum()
    n_clusters = len(resid)
    if n_clusters < 2:
        return np.nan
    scale = n_clusters / (n_clusters - 1)
    return float(np.sqrt(scale * (resid**2).sum()) / len(h))


def _fit(train, feats, rounds=None, valid=None):
    """Huấn luyện. Có `valid` thì dừng sớm; không thì chạy đúng `rounds` vòng."""
    dtrain = lgb.Dataset(train[feats], train["y"])
    if valid is None:
        return lgb.train(PARAMS, dtrain, num_boost_round=rounds)
    return lgb.train(
        PARAMS,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[lgb.Dataset(valid[feats], valid["y"], reference=dtrain)],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )


def embargoed(df, before_dn):
    """Chỉ giữ dòng mà cửa sổ nhãn HORIZON phiên đã kết thúc trước `before_dn`."""
    return df[df["dn"] + HORIZON <= before_dn]


def pick_rounds(train, feats):
    """Chọn số vòng boosting bằng dừng sớm trên quý cuối của tập train."""
    last_q = train["q"].max()
    valid = train[train["q"] == last_q]
    fit = embargoed(train[train["q"] < last_q], valid["dn"].min())
    if fit["y"].nunique() < 2 or valid["y"].nunique() < 2:
        return DEFAULT_ROUNDS
    model = _fit(fit, feats, valid=valid)
    return max(int(model.best_iteration or DEFAULT_ROUNDS), MIN_ROUNDS)


def walk_forward(df, feats, name, baseline_col):
    """Trả về dự đoán ngoài mẫu của mọi quý test và số vòng đã chọn."""
    quarters = sorted(df["q"].unique())
    parts, folds = [], []
    for i in range(MIN_TRAIN_QUARTERS, len(quarters)):
        test = df[df["q"] == quarters[i]]
        if test.empty:
            continue
        train = embargoed(df[df["q"] < quarters[i]], test["dn"].min())
        if train["y"].nunique() < 2 or test["y"].nunique() < 2:
            continue

        rounds = pick_rounds(train, feats)
        raw = _fit(train, feats, rounds=rounds).predict(test[feats])

        part = test[["t", "q", "g", "c", "y", "excess_fwd"]].copy()
        part["raw"] = raw
        part["behavior"] = normalize_within_group(test["g"], raw)
        part["baseline"] = test[baseline_col].fillna(0).to_numpy()
        parts.append(part)
        auc = float(roc_auc_score(test["y"], raw))
        folds.append({"quy": str(quarters[i]), "rounds": rounds, "auc": round(auc, 4)})
        log(
            "s5",
            f"[{name}] {quarters[i]} train={len(train):,} test={len(test):,} "
            f"vòng={rounds} AUC={auc:.3f}",
        )
    oof = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return oof, folds


def final_score(oof, kind, market_weight=MARKET_WEIGHT):
    if kind == "hybrid":
        return market_weight * oof["baseline"] + (1 - market_weight) * oof["behavior"]
    if kind == "research_first":
        # SELL > BUY > chưa có tín hiệu; behavior chỉ phá hoà bên trong cùng một
        # bậc và không thể lật quyết định của research.
        return oof["baseline"] + RESEARCH_TIEBREAK * oof["behavior"]
    return oof["behavior"]


def _metrics(oof, score):
    return ranking_metrics(
        oof["g"], oof["y"], oof["excess_fwd"], score, cluster=oof["c"]
    )


def evaluate(oof, folds, kind):
    """Bảng theo quý + số tổng hợp gộp mọi nhóm, cho 3 cách chấm điểm."""
    variants = {
        "final": final_score(oof, kind),
        "behavior": oof["behavior"],
        "baseline": oof["baseline"],
    }
    rows = []
    for fold in folds:
        mask = (oof["q"].astype(str) == fold["quy"]).to_numpy()
        part = oof[mask]
        row = dict(fold, n_test=len(part))
        for vname, score in variants.items():
            m = _metrics(part, score[mask])
            row["n_groups"] = m["n_groups"]
            row["random_hit@1"] = m["random_hit@1"]
            row |= {f"{vname}_{key}": m[key] for key in REPORT_KEYS}
        rows.append(row)

    pooled = {v: _metrics(oof, score) for v, score in variants.items()}
    return pd.DataFrame(rows), pooled


def weight_sweep(oof):
    """Đo lại điểm cuối BUY ở nhiều mức MARKET_WEIGHT trên cùng dự đoán ngoài mẫu."""
    rows = []
    for w in MARKET_WEIGHT_GRID:
        m = _metrics(oof, final_score(oof, "hybrid", market_weight=w))
        rows.append({"market_weight": w} | {key: m[key] for key in REPORT_KEYS})
    return pd.DataFrame(rows)


def save(model, df, feats, name, rounds):
    model.save_model(f"{OUT_DIR}/model_{name}.txt")
    meta = {
        "features": feats,
        "cats": [c for c in CATS if c in feats],
        "categories": {
            c: [str(v) for v in df[c].cat.categories] for c in CATS if c in feats
        },
        "rounds": rounds,
    }
    with open(f"{OUT_DIR}/meta_{name}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    imp = pd.DataFrame(
        {
            "feature": feats,
            "gain": model.feature_importance("gain"),
        }
    ).sort_values("gain", ascending=False)
    imp.to_csv(f"{OUT_DIR}/importance_{name}.csv", index=False)
    return imp


def load_table(table):
    """Đọc bảng train rồi đóng DuckDB ngay để nhường RAM cho LightGBM.

    Việc nặng làm trong DuckDB (có tràn ra đĩa) chứ không làm trong pandas:
      * sắp xếp cố định theo (t, khách, mã) — thứ tự dòng sau phép join bị lệch
        theo nhãn, nếu không sắp thì khi các mã bằng điểm nhau, thứ hạng sẽ vô
        tình "nhìn thấy" nhãn;
      * đánh số nhóm (khách, mốc) thành cột `g` và khách thành cột `c`, nên không
        cần mang hai cột chuỗi customer_id / stock_code sang pandas;
      * số thực trả về dạng FLOAT 4 byte, chuỗi trả về dạng từ điển (category).
    Nhờ vậy bảng danh mục chiếm khoảng 0,5 GB thay vì hơn 2 GB.
    """
    con = connect(read_only=True)
    try:
        select = ["CAST(t AS TIMESTAMP) AS t"]
        for name, dtype, *_ in con.execute(f"DESCRIBE {table}").fetchall():
            if name in ("t", "customer_id", "stock_code"):
                continue
            if dtype == "VARCHAR" or name == "dn":  # dn: số nguyên để chia train/test
                select.append(f'"{name}"')
            else:
                select.append(f'CAST("{name}" AS FLOAT) AS "{name}"')
        # Giữ các mốc mới nhất trong bảng feature để app vẫn chấm điểm được, nhưng
        # tuyệt đối không dùng chúng làm nhãn 0 khi chưa đi hết HORIZON phiên.
        table_arrow = con.execute(f"""
            SELECT {", ".join(select)},
                   CAST(DENSE_RANK() OVER (ORDER BY t, customer_id) - 1 AS INTEGER) AS g,
                   CAST(DENSE_RANK() OVER (ORDER BY customer_id) - 1 AS INTEGER) AS c
            FROM {table} WHERE label_complete = 1
            ORDER BY t, customer_id, stock_code""").arrow()
    finally:
        con.close()
    df = table_arrow.to_pandas(
        strings_to_categorical=True, split_blocks=True, self_destruct=True
    )
    del table_arrow
    df["q"] = pd.PeriodIndex(df["t"], freq="Q")
    return df


def _round_floats(d, digits=4):
    return {
        k: (round(v, digits) if isinstance(v, float) and np.isfinite(v) else v)
        for k, v in d.items()
        if not (isinstance(v, float) and not np.isfinite(v))
    }


def train_one(table, name, kind, extra_drop, baseline_col):
    log("s5", f"=== MODEL {name.upper()} ===")
    df = load_table(table)
    ablated = sorted(
        c for c in df.columns if any(fnmatch.fnmatch(c, p) for p in ABLATE_FEATURES)
    )
    if ablated:
        log("s5", f"PHÉP THỬ: bỏ {len(ablated)} đặc trưng — {', '.join(ablated)}")
    df, feats = prep(df, [*extra_drop, *ablated])
    # Luật cứng: MarketScore và thứ hạng research không được vào model hành vi. Chặn
    # theo tên cột chứ không chỉ dựa vào danh sách loại, để một bảng train dựng bởi
    # phiên bản cũ (còn cột lạ) làm hỏng lần chạy thay vì âm thầm rò rỉ.
    leaked = [f for f in feats if f.startswith(RESEARCH_COLUMN_PREFIXES)]
    if leaked:
        sys.exit(f"[{name}] cột research lọt vào đặc trưng: {', '.join(leaked)}")
    log(
        "s5",
        f"{len(df):,} dòng | {len(feats)} đặc trưng | "
        f"nhãn dương {100 * df.y.mean():.2f}%",
    )

    oof, folds = walk_forward(df, feats, name, baseline_col)
    result = None
    if folds:
        evaluation, pooled = evaluate(oof, folds, kind)
        evaluation.to_csv(f"{OUT_DIR}/eval_{name}.csv", index=False)
        result = {
            "auc": round(float(np.mean([f["auc"] for f in folds])), 4),
            "n_folds": len(folds),
            "n_groups": pooled["final"]["n_groups"],
            "n_groups_all": pooled["final"]["n_groups_all"],
            "random_hit@1": round(pooled["final"]["random_hit@1"], 4),
            **{v: _round_floats(m) for v, m in pooled.items()},
        }
        if kind == "hybrid":
            sweep = weight_sweep(oof)
            sweep.to_csv(f"{OUT_DIR}/eval_{name}_weights.csv", index=False)
            result["weights"] = [_round_floats(r) for r in sweep.to_dict("records")]
        f, b, r = pooled["final"], pooled["behavior"], pooled["baseline"]
        log(
            "s5",
            f"[{name}] {f['n_groups']:,} nhóm xếp hạng được | HitRate@1: "
            f"ngẫu nhiên {f['random_hit@1']:.3f} | baseline {r['hit@1']:.3f} | "
            f"hành vi {b['hit@1']:.3f} | điểm cuối {f['hit@1']:.3f}",
        )

    rounds = int(np.median([f["rounds"] for f in folds])) if folds else DEFAULT_ROUNDS
    final_model = _fit(df, feats, rounds=rounds)
    importance = save(final_model, df, feats, name, rounds)
    if result is not None:
        result["rounds"] = rounds
    log(
        "s5",
        f"[{name}] model cuối {rounds} vòng | 8 đặc trưng mạnh nhất: "
        f"{', '.join(importance.head(8).feature)}",
    )
    return result


def _mean_se(values, overlap):
    """Trung bình và sai số chuẩn khi các quan sát liền nhau chồng cửa sổ.

    `overlap` = số quan sát cùng nằm trong một cửa sổ HORIZON phiên. Số quan sát
    độc lập thực tế chỉ còn n / overlap, nên sai số chuẩn được nới ra tương ứng.
    """
    v = pd.Series(values, dtype=float).dropna()
    if len(v) < 2:
        return {"n": len(v)}
    n_eff = max(len(v) / overlap, 1.0)
    se = float(v.std(ddof=1) / np.sqrt(n_eff))
    return {
        "n": len(v),
        "n_independent": round(n_eff, 1),
        "mean": round(float(v.mean()), 4),
        "se": round(se, 4),
        "t_stat": round(float(v.mean() / se), 2) if se > 0 else None,
        "share_positive": round(float((v > 0).mean()), 3),
    }


def audit_market_score():
    """MarketScore có dự báo lợi suất không? Đo trên chính rổ PHS BUY mới trong ngày.

    Mỗi mốc là một quan sát (không phải mỗi khách — mọi khách thấy cùng một rổ).
    Dùng mọi mốc để trung bình không phụ thuộc việc chọn mốc nào, nhưng sai số
    chuẩn tính theo số cửa sổ HORIZON phiên KHÔNG chồng nhau; nếu không, nhịp
    tuần sẽ cho sai số nhỏ giả tạo.
      * basket_excess: lợi suất vượt VNINDEX trung bình của cả rổ PHS.
      * top_minus_basket: mã MarketScore cao nhất so với trung bình rổ.
      * rank_ic: tương quan hạng giữa MarketScore và lợi suất (rổ >= 3 mã).
    """
    con = connect(read_only=True)
    try:
        d = con.execute("""
            SELECT t, dn, stock_code, market_score, excess_fwd FROM stock_at_t
            WHERE research_recommendation = 'BUY' AND label_complete = 1
              AND market_score IS NOT NULL AND excess_fwd IS NOT NULL
            ORDER BY dn, stock_code""").df()
        gaps = con.execute(
            "SELECT MEDIAN(gap) FROM (SELECT dn - LAG(dn) OVER (ORDER BY dn) AS gap "
            "FROM decision_dates)"
        ).fetchone()[0]
    finally:
        con.close()
    if d.empty:
        return None

    rows = []
    for _, x in d.groupby("dn", sort=True):
        row = {"basket": float(x["excess_fwd"].mean()), "top": np.nan, "ic": np.nan}
        if len(x) >= 2:
            top = x.loc[x["market_score"].idxmax(), "excess_fwd"]
            row["top"] = float(top) - row["basket"]
        if len(x) >= 3:
            row["ic"] = x["market_score"].rank().corr(x["excess_fwd"].rank())
        rows.append(row)
    r = pd.DataFrame(rows)
    overlap = max(HORIZON / float(gaps or HORIZON), 1.0)
    return {
        "horizon": HORIZON,
        "basket_excess": _mean_se(r["basket"], overlap),
        "top_minus_basket": _mean_se(r["top"], overlap),
        "rank_ic": _mean_se(r["ic"], overlap),
    }


def check_ablation():
    """Dừng ngay nếu một mẫu trong ABLATE_FEATURES không khớp cột nào. Gõ nhầm tên
    (peers_* thay vì peer_*) mà vẫn chạy thì sẽ ra kết luận sai "bỏ đi không ảnh hưởng"."""
    if not ABLATE_FEATURES:
        return
    con = connect(read_only=True)
    try:
        columns = {
            r[0]
            for table in ("train_buy", "train_portfolio")
            for r in con.execute(f"DESCRIBE {table}").fetchall()
        }
    finally:
        con.close()
    unmatched = [
        p for p in ABLATE_FEATURES if not any(fnmatch.fnmatch(c, p) for c in columns)
    ]
    if unmatched:
        sys.exit(
            "ABLATE_FEATURES có mẫu không khớp đặc trưng nào: " + ", ".join(unmatched)
        )


def main():
    check_ablation()
    # MarketScore bị loại khỏi behavior model để hai thành phần độc lập; nó chỉ
    # được ghép lại bằng MARKET_WEIGHT khi xếp hạng.
    summary = {
        "market_weight": MARKET_WEIGHT,
        "decision_freq": DECISION_FREQ,
        "eval_ks": list(EVAL_KS),
        "sample_pct": SAMPLE_PCT,
        "graph_backend": GRAPH_BACKEND,
        "ablate_features": ABLATE_FEATURES,
    }
    summary["buy"] = train_one(
        "train_buy",
        "buy",
        kind="hybrid",
        extra_drop=[
            "market_score",
            "candidate_rank",
            "market_score_norm",
            "target_price",
            "cut_loss_price",
        ],
        baseline_col="market_score_norm",
    )
    summary["portfolio"] = train_one(
        "train_portfolio",
        "portfolio",
        kind="research_first",
        extra_drop=[
            "market_score",
            "candidate_rank",
            "research_recommendation",
            "research_call_type",
            "in_research",
            "research_sell_signal",
        ],
        baseline_col="research_sell_signal",
    )

    summary["market_score_audit"] = audit_market_score()
    if summary["market_score_audit"]:
        a = summary["market_score_audit"]
        log(
            "s5",
            f"audit MarketScore ({a['basket_excess']['n']} mốc): rổ PHS vượt VNINDEX "
            f"{a['basket_excess'].get('mean')} (t={a['basket_excess'].get('t_stat')}) | "
            f"mã điểm cao nhất so với rổ {a['top_minus_basket'].get('mean')} "
            f"(t={a['top_minus_basket'].get('t_stat')})",
        )

    with open(f"{OUT_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log("s5", "xong.")


if __name__ == "__main__":
    main()
