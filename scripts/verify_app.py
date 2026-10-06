"""Kiểm tra hệ thống THẬT đang chạy — gọi qua `scripts/verify.sh full`.

Chạy bên trong container app (đọc từ stdin), chỉ mở DuckDB ở chế độ chỉ-đọc nên
không thay đổi gì. Trình tự:

  Doctor   pipeline đã chạy đúng phiên bản, bảng đủ dữ liệu, vị thế khớp snapshot,
           đồ thị Neo4j khớp đúng bộ dữ liệu hiện tại, model thắng chọn ngẫu nhiên.
  Drive    gọi hàm gợi ý của app cho một khách thật, tại một mốc có rổ PHS và một
           mốc không có rổ.
  Evidence đọc lại dữ liệu gốc để đối chiếu với thứ app trả về, rồi ghi tất cả vào
           artifacts/verify_report.json.

Thoát mã 1 nếu có bất kỳ kiểm tra nào không qua.
"""

import datetime as dt
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, "/app/src")

from neo4j import GraphDatabase

import app
import pipeline
import s3_graph
from config import (
    GRADIO_PORT,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OUT_DIR,
    TOPK,
    connect,
)

# model phải hơn chọn ngẫu nhiên ít nhất chừng này về HitRate@1
MIN_LIFT_OVER_RANDOM = 0.02
# số khách được gọi gợi ý rồi đối chiếu với dữ liệu gốc
N_CUSTOMERS = int(os.getenv("VERIFY_CUSTOMERS", "20"))
REQUIRED_TABLES = [
    "txn",
    "px",
    "decision_dates",
    "phs_calls",
    "research",
    "pos",
    "cust_pit",
    "cust_horizon",
    "graph_pop",
    "graph_cooc",
    "graph_peer",
    "train_buy",
    "train_portfolio",
]

checks = []


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    print(f"  [{'OK ' if ok else 'LỖI'}] {name}" + (f" — {detail}" if detail else ""))


def one(con, sql, params=None):
    return con.execute(sql, params or []).fetchone()[0]


def doctor(con):
    print("Doctor")
    try:
        with open(pipeline.DONE, encoding="utf-8") as f:
            done = f.read().strip().split("|")[0]
    except OSError:
        done = "(không có)"
    check(
        "pipeline đã chạy đúng phiên bản hiện tại",
        done == pipeline.PIPELINE_VERSION,
        f"{done} / cần {pipeline.PIPELINE_VERSION}",
    )

    empty = [t for t in REQUIRED_TABLES if one(con, f"SELECT COUNT(*) FROM {t}") == 0]
    check(
        "mọi bảng chính đều có dữ liệu",
        not empty,
        f"bảng rỗng: {empty}" if empty else "",
    )

    n, same = con.execute("""
        SELECT COUNT(*),
               COUNT_IF(s.qty = h.qty
                        AND ABS(s.avg_cost - h.avg_cost) <= 1e-4 * h.avg_cost)
        FROM hold h
        ASOF LEFT JOIN pos_state s
               ON s.customer_id = h.customer_id AND s.stock_code = h.stock_code
              AND h.t >= s.d
    """).fetchone()
    check(
        "vị thế dựng lại khớp snapshot cuối tháng",
        n > 0 and same == n,
        f"{same:,}/{n:,}",
    )

    bad = one(
        con,
        """
        SELECT COUNT(*) FROM train_buy b
        WHERE NOT EXISTS (
            SELECT 1 FROM phs_calls pc
            WHERE pc.stock_code = b.stock_code AND pc.open_d = b.t)""",
    )
    check("mọi ứng viên MUA là mã PHS mới BUY đúng ngày t", bad == 0, f"{bad:,} dòng sai")

    drv = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), notifications_min_severity="OFF"
    )
    try:
        with drv.session() as s:
            rec = s.run(
                "MATCH (m:Meta {key: 'graph'}) RETURN m.fingerprint AS fp"
            ).single()
            edges = s.run("MATCH ()-[r:TRADED]->() RETURN count(r) AS n").single()["n"]
    finally:
        drv.close()
    n_txn = one(con, "SELECT COUNT(*) FROM txn")
    check(
        "đồ thị Neo4j khớp đúng bộ dữ liệu hiện tại",
        rec is not None
        and rec["fp"] == s3_graph.data_fingerprint(con)
        and edges == n_txn,
        f"{edges:,} cạnh / {n_txn:,} giao dịch",
    )

    with open(f"{OUT_DIR}/summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    metrics = {}
    for name in ("buy", "portfolio"):
        r = summary.get(name) or {}
        behavior = (r.get("behavior") or {}).get("hit@1")
        random_hit = r.get("random_hit@1")
        metrics[name] = {
            "auc": r.get("auc"),
            "random_hit@1": random_hit,
            "behavior_hit@1": behavior,
            "final_hit@1": (r.get("final") or {}).get("hit@1"),
            "n_groups": r.get("n_groups"),
        }
        ok = (
            behavior is not None
            and random_hit is not None
            and behavior >= random_hit + MIN_LIFT_OVER_RANDOM
        )
        check(
            f"model {name} hơn chọn ngẫu nhiên (HitRate@1)",
            ok,
            f"hành vi {behavior} / ngẫu nhiên {random_hit}",
        )
    return metrics


def percent_ok(series):
    values = series.astype(str).str.rstrip("%").astype(float)
    return bool(((values >= 0) & (values <= 100)).all())


def drive(con):
    print("Drive + Evidence")
    ui = app.build()

    t = one(
        con,
        """
        SELECT t FROM train_buy GROUP BY t
        HAVING COUNT(DISTINCT stock_code) >= 2 ORDER BY t DESC LIMIT 1""",
    )
    # Mẫu khách cố định (chọn theo băm mã khách) trong số khách đang có vị thế tại t.
    customers = [
        r[0]
        for r in con.execute(
            """
            SELECT customer_id FROM train_portfolio WHERE t = ?
            GROUP BY 1 HAVING COUNT(*) >= 2
            ORDER BY HASH(customer_id) LIMIT ?""",
            [t, N_CUSTOMERS],
        ).fetchall()
    ]
    # --- dữ liệu gốc để đối chiếu: khuyến nghị PHS mới trong ngày, và vị thế của từng khách
    buy_calls = {
        r[0]
        for r in con.execute(
            """
            SELECT stock_code FROM phs_calls
            WHERE open_d = ?""",
            [t],
        ).fetchall()
    }
    order = {"BÁN": 0, "CÓ THỂ MUA": 1, "GIỮ": 2, "CHƯA CÓ TÍN HIỆU": 3}
    passed = dict.fromkeys(
        (
            "answered",
            "buy_open",
            "buy_sorted",
            "buy_prob",
            "held",
            "sell_first",
            "sell_prob",
        ),
        0,
    )
    example = None
    for cid in customers:
        header, buy, port = ui.recommend(cid, str(t))
        held = {
            r[0]
            for r in con.execute(
                "SELECT stock_code FROM pos WHERE customer_id = ? AND t = ?", [cid, t]
            ).fetchall()
        }
        shown, scores = list(buy["Mã"]), list(buy["Điểm cuối"])
        ranks = [order.get(a, 99) for a in port["Khuyến nghị"]]
        passed["answered"] += cid in header and str(t) in header
        passed["buy_open"] += bool(shown) and set(shown) <= buy_calls
        passed["buy_sorted"] += len(shown) <= TOPK and scores == sorted(
            scores, reverse=True
        )
        passed["buy_prob"] += percent_ok(buy["Khả năng khách mua (model)"])
        passed["held"] += bool(set(port["Mã"])) and set(port["Mã"]) <= held
        passed["sell_first"] += ranks == sorted(ranks)
        passed["sell_prob"] += percent_ok(port["Khả năng khách bán (model)"])
        if example is None:
            example = {
                "customer_id": cid,
                "t": str(t),
                "buy": buy.astype(str).to_dict("records"),
                "portfolio_top": port.head(5).astype(str).to_dict("records"),
            }

    n = len(customers)
    labels = {
        "answered": "app trả lời đúng khách và mốc được hỏi",
        "buy_open": "mã MUA hiển thị đều là khuyến nghị PHS mới trong ngày (đọc lại phs_calls)",
        "buy_sorted": "nhánh MUA: không quá TOPK mã, xếp theo điểm cuối giảm dần",
        "buy_prob": "khả năng khách mua là xác suất hợp lệ",
        "held": "danh mục hiển thị đúng các mã khách đang nắm (đọc lại pos)",
        "sell_first": "danh mục: tín hiệu BÁN của PHS luôn đứng trước",
        "sell_prob": "khả năng khách bán là xác suất hợp lệ",
    }
    for key, label in labels.items():
        check(label, n > 0 and passed[key] == n, f"{passed[key]}/{n} khách @ {t}")
    cid = customers[0] if customers else None

    # --- mốc không có rổ PHS: phải báo rõ, không được tự bịa mã
    empty_t = con.execute("""
        SELECT dd.t FROM decision_dates dd
        WHERE NOT EXISTS (SELECT 1 FROM train_buy b WHERE b.t = dd.t)
        ORDER BY dd.t DESC LIMIT 1""").fetchone()
    if empty_t:
        _, buy_empty, _ = ui.recommend(cid, str(empty_t[0]))
        check(
            "mốc không có khuyến nghị PHS: app báo rõ thay vì gợi ý mã",
            list(buy_empty.columns) == ["Thông báo"],
            f"mốc {empty_t[0]}",
        )

    # Chạy từ container khác (Airflow) thì trỏ VERIFY_APP_URL về service app.
    url = os.getenv("VERIFY_APP_URL", f"http://localhost:{GRADIO_PORT}/")
    status, deadline = None, time.time() + int(os.getenv("VERIFY_APP_WAIT", "0"))
    while True:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                status = r.status
        except OSError as e:
            status = str(e)
        if status == 200 or time.time() >= deadline:
            break
        time.sleep(5)
    check("giao diện Gradio trả HTTP 200", status == 200, f"{url} -> {status}")

    return example


def main():
    con = connect(read_only=True)
    try:
        metrics = doctor(con)
        example = drive(con)
    finally:
        con.close()

    failed = [c for c in checks if not c["ok"]]
    report = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": pipeline.PIPELINE_VERSION,
        "passed": not failed,
        "checks": checks,
        "metrics": metrics,
        "example": example,
    }
    path = f"{OUT_DIR}/verify_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"\n{len(checks) - len(failed)}/{len(checks)} kiểm tra qua — bằng chứng: {path}"
    )
    for name, m in metrics.items():
        print(f"  {name}: {m}")
    if failed:
        print("KHÔNG QUA: " + "; ".join(c["name"] for c in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
