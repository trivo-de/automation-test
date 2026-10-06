"""
Bước 3 — Neo4j
Nạp đồ thị Khách - Mã - Ngành, rồi rút ra đặc trưng đồ thị THEO LÁT CẮT THỜI GIAN.

Nguyên tắc chống rò rỉ (quan trọng nhất của bước này):
  Cạnh TRADED mang sẵn thuộc tính ngày `d`. Mọi truy vấn đặc trưng đều có
  `WHERE r.d < $t`, nên đặc trưng tại t chỉ nhìn thấy lịch sử trước t.
  Không cần dựng lại đồ thị cho từng mốc, không cần embedding.

Đặc trưng lấy ra:
  1. pop(a)      — số khách đã mua / bán mã a trong cửa sổ gần đây
  2. cooc(a, b)  — số khách cùng mua cả mã a và mã b (đường đi Mã <- Khách -> Mã)
  3. peer(c, p)  — các khách p giống khách c nhất theo rổ mã đã mua
                   (đường đi Khách -> Mã <- Khách, độ giống Jaccard, giữ top-K).
     Đây là phép duyệt 2 bước trên đồ thị; s4 nối thêm bước thứ 3
     (khách tương tự -> mã họ vừa mua/bán) để cá nhân hoá cho từng khách.
Tính tại mốc cuối THÁNG rồi gán tới mọi mốc quyết định bằng as-of join.
"""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from neo4j import GraphDatabase

from config import (
    COOC_WINDOW,
    GRAPH_WORKERS,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PEER_MIN_SHARED,
    PEER_TOPK,
    connect,
    log,
)

BATCH = 10_000


def data_fingerprint(con) -> str:
    """Dấu vân tay của dữ liệu đã nạp vào đồ thị.

    Chỉ so số cạnh là không đủ: thay bộ dữ liệu khác có cùng số dòng giao dịch sẽ
    bị coi là "đã nạp". Băm nội dung từng dòng thì mọi thay đổi đều bị phát hiện.
    """
    parts = con.execute("""
        SELECT
          (SELECT COUNT(*) || ':' || COALESCE(BIT_XOR(HASH(CONCAT_WS('|',
                  customer_id, stock_code, CAST(d AS VARCHAR), side,
                  CAST(qty AS VARCHAR), CAST(value AS VARCHAR)))), 0) FROM txn),
          (SELECT COUNT(*) || ':' || COALESCE(BIT_XOR(HASH(CONCAT_WS('|',
                  stock_code, exchange, icb_code, icb_name))), 0) FROM sec),
          (SELECT COUNT(*) || ':' || COALESCE(BIT_XOR(HASH(CONCAT_WS('|',
                  customer_id, customer_type))), 0) FROM prof)
    """).fetchone()
    return "txn=" + parts[0] + ";sec=" + parts[1] + ";prof=" + parts[2]


def wipe_graph(s):
    """Xoá sạch đồ thị theo lô để không tràn bộ nhớ transaction."""
    s.run("MATCH (m:Meta) DELETE m").consume()
    s.run(
        "MATCH ()-[r]->() CALL (r) { DELETE r } IN TRANSACTIONS OF 20000 ROWS"
    ).consume()
    s.run("MATCH (n) CALL (n) { DELETE n } IN TRANSACTIONS OF 20000 ROWS").consume()


def load_graph(drv, con):
    log("s3", "tạo ràng buộc & chỉ mục...")
    with drv.session() as s:
        s.run(
            "CREATE CONSTRAINT cust_id IF NOT EXISTS "
            "FOR (c:Customer) REQUIRE c.id IS UNIQUE"
        )
        s.run(
            "CREATE CONSTRAINT stock_code IF NOT EXISTS "
            "FOR (x:Stock) REQUIRE x.code IS UNIQUE"
        )
        s.run(
            "CREATE CONSTRAINT sector_code IF NOT EXISTS "
            "FOR (g:Sector) REQUIRE g.code IS UNIQUE"
        )
        s.run("CREATE INDEX traded_date IF NOT EXISTS FOR ()-[r:TRADED]-() ON (r.d)")

        fingerprint = data_fingerprint(con)
        rec = s.run("MATCH (m:Meta {key: 'graph'}) RETURN m.fingerprint AS fp").single()
        if rec and rec["fp"] == fingerprint:
            log("s3", "đồ thị đã khớp đúng bộ dữ liệu hiện tại — bỏ qua bước nạp.")
            return

        # Dấu vân tay chỉ được ghi SAU khi nạp xong, nên cả hai trường hợp "dữ liệu
        # đã đổi" và "lần nạp trước bị ngắt giữa chừng" đều đi vào nhánh này.
        n_old = s.run("MATCH ()-[r:TRADED]->() RETURN count(r) AS n").single()["n"]
        if n_old > 0 or rec:
            log(
                "s3",
                f"đồ thị cũ ({n_old:,} cạnh) không khớp dữ liệu hiện tại "
                "hoặc nạp dở — xoá và nạp lại.",
            )
            wipe_graph(s)

        log("s3", "nạp node Khách / Mã / Ngành...")
        cust = con.execute("SELECT customer_id, customer_type FROM prof").fetchall()
        s.run(
            "UNWIND $rows AS r MERGE (c:Customer {id: r[0]}) SET c.type=r[1]",
            rows=cust,
        )

        stk = con.execute(
            "SELECT stock_code, exchange, icb_code, icb_name FROM sec"
        ).fetchall()
        s.run(
            "UNWIND $rows AS r MERGE (x:Stock {code: r[0]}) "
            "SET x.exchange=r[1], x.icb=r[2]",
            rows=stk,
        )

        sect = con.execute("SELECT DISTINCT icb_code, icb_name FROM sec").fetchall()
        s.run(
            "UNWIND $rows AS r MERGE (g:Sector {code: r[0]}) SET g.name=r[1]", rows=sect
        )
        s.run(
            """UNWIND $rows AS r
                 MATCH (x:Stock {code:r[0]}), (g:Sector {code:r[1]})
                 MERGE (x)-[:IN_SECTOR]->(g)""",
            rows=con.execute("SELECT stock_code, icb_code FROM sec").fetchall(),
        )

        log("s3", "nạp cạnh TRADED (có mốc thời gian)...")
        rows = con.execute("""SELECT customer_id, stock_code, CAST(d AS VARCHAR),
                                     side, qty, value FROM txn ORDER BY d""").fetchall()
        for i in range(0, len(rows), BATCH):
            s.run(
                """UNWIND $rows AS r
                     MATCH (c:Customer {id: r[0]}), (x:Stock {code: r[1]})
                     CREATE (c)-[:TRADED {d: r[2], side: r[3], qty: r[4],
                                          value: r[5]}]->(x)""",
                rows=rows[i : i + BATCH],
            )
            if (i // BATCH) % 10 == 0:
                log("s3", f"   ... {min(i + BATCH, len(rows)):,}/{len(rows):,} cạnh")

        n_new = s.run("MATCH ()-[r:TRADED]->() RETURN count(r) AS n").single()["n"]
        if n_new != len(rows):
            raise RuntimeError(
                f"nạp đồ thị thiếu cạnh: {n_new:,}/{len(rows):,} — giao dịch có mã "
                "hoặc khách không nằm trong danh mục?"
            )
        s.run(
            "MERGE (m:Meta {key: 'graph'}) SET m.fingerprint = $fp", fp=fingerprint
        ).consume()
        log("s3", f"nạp xong {len(rows):,} cạnh.")


POP_QUERY = """
MATCH (c:Customer)-[r:TRADED]->(x:Stock)
WHERE r.d > $t0 AND r.d < $t
RETURN x.code AS stock_code,
       count(DISTINCT CASE WHEN r.side='BUY' THEN c END) AS n_buyers,
       sum(CASE WHEN r.side='BUY' THEN r.value ELSE 0 END) AS buy_value,
       count(DISTINCT CASE WHEN r.side='SELL' THEN c END) AS n_sellers,
       sum(CASE WHEN r.side='SELL' THEN r.value ELSE 0 END) AS sell_value"""

COOC_QUERY = """
MATCH (c:Customer)-[r1:TRADED]->(a:Stock)
WHERE r1.side='BUY' AND r1.d > $t0 AND r1.d < $t
MATCH (c)-[r2:TRADED]->(b:Stock)
WHERE r2.side='BUY' AND r2.d > $t0 AND r2.d < $t
  AND elementId(a) < elementId(b)
WITH a.code AS a, b.code AS b, count(DISTINCT c) AS co
WHERE co >= 3
RETURN a, b, co"""

# Khách -> Mã <- Khách: hai khách càng mua chung nhiều mã (so với tổng số mã mỗi
# người mua) thì càng giống nhau. Gom người mua theo từng mã trước rồi mới ghép
# cặp, để một khách mua một mã nhiều lần không làm nổ số đường đi.
PEER_QUERY = """
MATCH (c:Customer)-[r:TRADED]->(s:Stock)
WHERE r.side='BUY' AND r.d > $t0 AND r.d < $t
WITH c, collect(DISTINCT s) AS stocks
WITH c, stocks, size(stocks) AS deg
UNWIND stocks AS s
WITH s, collect({c: c, deg: deg}) AS buyers
UNWIND buyers AS x
UNWIND buyers AS y
WITH x, y WHERE x.c <> y.c
WITH x.c AS c1, y.c AS c2, x.deg AS d1, y.deg AS d2, count(*) AS shared
WHERE shared >= $min_shared
WITH c1, c2, shared * 1.0 / (d1 + d2 - shared) AS sim
ORDER BY sim DESC, c2.id
WITH c1, collect({peer: c2.id, sim: sim})[..$k] AS peers
UNWIND peers AS p
RETURN c1.id AS customer_id, p.peer AS peer_id, p.sim AS sim"""


def extract_features(drv, con):
    """Tính đặc trưng đồ thị tại từng mốc cuối tháng — luôn có WHERE r.d < t."""
    slices = [
        r[0]
        for r in con.execute("""
        SELECT MAX(t) FROM decision_dates
        GROUP BY DATE_TRUNC('month', t) ORDER BY 1""").fetchall()
    ]
    log(
        "s3",
        f"tính đặc trưng đồ thị tại {len(slices)} mốc cuối tháng "
        f"(cửa sổ {COOC_WINDOW} ngày)...",
    )

    con.execute("""
    CREATE OR REPLACE TABLE graph_pop (
        qt DATE, stock_code VARCHAR, n_buyers BIGINT, buy_value DOUBLE,
        n_sellers BIGINT, sell_value DOUBLE);
    CREATE OR REPLACE TABLE graph_cooc (qt DATE, a VARCHAR, b VARCHAR, co BIGINT);
    CREATE OR REPLACE TABLE graph_peer (
        qt DATE, customer_id VARCHAR, peer_id VARCHAR, sim DOUBLE);
    """)

    def append(table, df, columns):
        """Ghi từng lát cắt vào DuckDB ngay, không giữ mọi lát cắt trong RAM."""
        if df.empty:
            return
        con.register("_graph_df", df)
        con.execute(
            f"INSERT INTO {table} SELECT CAST(qt AS DATE), {columns} FROM _graph_df"
        )
        con.unregister("_graph_df")

    def query_slice(q):
        """Ba truy vấn của một lát cắt, trong session riêng (chạy được song song)."""
        t = str(q)
        t0 = str(q - dt.timedelta(days=COOC_WINDOW))
        with drv.session() as s:
            pop = pd.DataFrame(s.run(POP_QUERY, t0=t0, t=t).data())
            co = pd.DataFrame(s.run(COOC_QUERY, t0=t0, t=t).data())
            peer = pd.DataFrame(
                s.run(
                    PEER_QUERY, t0=t0, t=t, k=PEER_TOPK, min_shared=PEER_MIN_SHARED
                ).data()
            )
        return t, pop, co, peer

    # Neo4j Community chạy mỗi truy vấn trên một nhân, nên gửi vài lát cắt cùng lúc
    # để dùng nhiều nhân. Ghi vào DuckDB vẫn tuần tự ở luồng chính, theo đúng thứ tự.
    n_pop = n_cooc = n_peer = 0
    with ThreadPoolExecutor(max_workers=GRAPH_WORKERS) as pool:
        for i, (t, pop, co, peer) in enumerate(pool.map(query_slice, slices)):
            if not pop.empty:
                pop.insert(0, "qt", t)
                append(
                    "graph_pop",
                    pop,
                    "stock_code, n_buyers, buy_value, n_sellers, sell_value",
                )
            if not co.empty:
                # đồng mua là quan hệ hai chiều -> ghi cả 2 hướng cho dễ join
                both = pd.concat(
                    [co, co.rename(columns={"a": "b", "b": "a"})], ignore_index=True
                )
                both.insert(0, "qt", t)
                append("graph_cooc", both, "a, b, co")
            if not peer.empty:
                peer.insert(0, "qt", t)
                append("graph_peer", peer, "customer_id, peer_id, sim")

            n_pop += len(pop)
            n_cooc += 2 * len(co)
            n_peer += len(peer)
            if i % 6 == 0 or i == len(slices) - 1:
                log(
                    "s3",
                    f"   {t}: {len(pop)} mã, {len(co):,} cặp đồng mua, "
                    f"{len(peer):,} cặp khách tương tự",
                )

    # gán lát cắt gần nhất <= t cho từng mốc quyết định (as-of, giữ point-in-time)
    con.execute("""
    CREATE OR REPLACE TABLE dd_q AS
    SELECT d.t, (SELECT MAX(qt) FROM graph_pop g WHERE g.qt <= d.t) AS qt
    FROM decision_dates d;
    """)
    log(
        "s3",
        f"graph_pop {n_pop:,} dòng | graph_cooc {n_cooc:,} dòng | "
        f"graph_peer {n_peer:,} dòng",
    )


def main():
    con = connect()
    # Tắt thông báo của server (nhãn chưa tồn tại ở lần chạy đầu, v.v.) để log gọn.
    drv = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        notifications_min_severity="OFF",
    )
    drv.verify_connectivity()
    load_graph(drv, con)
    extract_features(drv, con)
    drv.close()
    con.close()
    log("s3", "xong.")


if __name__ == "__main__":
    main()
