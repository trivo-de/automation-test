"""
Bước 3 (bản SQL) — tính đúng ba bảng đồ thị của s3_graph bằng DuckDB, không cần Neo4j.

Dùng cho chế độ chạy nhanh trên mẫu khách (GRAPH_BACKEND=sql) và cho bộ test. Pipeline
chính vẫn dùng Neo4j; hai bản phải cho cùng kết quả trên cùng dữ liệu, và điều đó được
đối chiếu bằng scripts/compare_graph.py.

Giữ nguyên ngữ nghĩa của các truy vấn Cypher:
  * cửa sổ: t - COOC_WINDOW < ngày giao dịch < t (không bao giờ nhìn thấy ngày t trở đi);
  * cooc: chỉ giữ cặp mã có ít nhất 3 khách cùng mua, ghi cả hai chiều;
  * peer: Jaccard trên rổ mã đã mua, ít nhất PEER_MIN_SHARED mã chung, mỗi khách giữ
    PEER_TOPK người giống nhất (hoà thì xếp theo mã khách).
"""

from config import COOC_WINDOW, PEER_MIN_SHARED, PEER_TOPK, connect, log

MIN_COBUYERS = 3


def build(con):
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE _slices AS
    SELECT MAX(t) AS qt FROM decision_dates GROUP BY DATE_TRUNC('month', t);

    CREATE OR REPLACE TABLE graph_pop AS
    SELECT s.qt, x.stock_code,
           COUNT(DISTINCT CASE WHEN x.side = 'BUY' THEN x.customer_id END) AS n_buyers,
           SUM(CASE WHEN x.side = 'BUY' THEN x.value ELSE 0 END) AS buy_value,
           COUNT(DISTINCT CASE WHEN x.side = 'SELL' THEN x.customer_id END) AS n_sellers,
           SUM(CASE WHEN x.side = 'SELL' THEN x.value ELSE 0 END) AS sell_value
    FROM _slices s
    JOIN txn x ON x.d < s.qt AND x.d > s.qt - {COOC_WINDOW}
    GROUP BY 1, 2;

    -- rổ mã mỗi khách đã mua trong cửa sổ của từng lát cắt
    CREATE OR REPLACE TEMP TABLE _bought AS
    SELECT DISTINCT s.qt, x.customer_id, x.stock_code
    FROM _slices s
    JOIN txn x ON x.d < s.qt AND x.d > s.qt - {COOC_WINDOW}
    WHERE x.side = 'BUY';

    CREATE OR REPLACE TABLE graph_cooc AS
    SELECT a.qt, a.stock_code AS a, b.stock_code AS b, COUNT(*) AS co
    FROM _bought a
    JOIN _bought b ON b.qt = a.qt AND b.customer_id = a.customer_id
                  AND b.stock_code <> a.stock_code
    GROUP BY 1, 2, 3
    HAVING COUNT(*) >= {MIN_COBUYERS};

    CREATE OR REPLACE TABLE graph_peer AS
    WITH deg AS (
        SELECT qt, customer_id, COUNT(*) AS deg FROM _bought GROUP BY 1, 2
    ),
    shared AS (
        SELECT a.qt, a.customer_id, b.customer_id AS peer_id, COUNT(*) AS shared
        FROM _bought a
        JOIN _bought b ON b.qt = a.qt AND b.stock_code = a.stock_code
                      AND b.customer_id <> a.customer_id
        GROUP BY 1, 2, 3
        HAVING COUNT(*) >= {PEER_MIN_SHARED}
    ),
    sim AS (
        SELECT s.qt, s.customer_id, s.peer_id,
               s.shared * 1.0 / (d1.deg + d2.deg - s.shared) AS sim
        FROM shared s
        JOIN deg d1 ON d1.qt = s.qt AND d1.customer_id = s.customer_id
        JOIN deg d2 ON d2.qt = s.qt AND d2.customer_id = s.peer_id
    )
    SELECT qt, customer_id, peer_id, sim
    FROM sim
    QUALIFY ROW_NUMBER() OVER (PARTITION BY qt, customer_id
                               ORDER BY sim DESC, peer_id) <= {PEER_TOPK};

    -- gán lát cắt gần nhất <= t cho từng mốc quyết định (as-of, giữ point-in-time)
    CREATE OR REPLACE TABLE dd_q AS
    SELECT d.t, (SELECT MAX(qt) FROM graph_pop g WHERE g.qt <= d.t) AS qt
    FROM decision_dates d;

    DROP TABLE _bought;
    DROP TABLE _slices;
    """)


def main():
    con = connect()
    log(
        "s3",
        f"tính đặc trưng đồ thị bằng SQL (cửa sổ {COOC_WINDOW} ngày, không Neo4j)...",
    )
    build(con)
    counts = [
        con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("graph_pop", "graph_cooc", "graph_peer")
    ]
    log(
        "s3",
        f"graph_pop {counts[0]:,} dòng | graph_cooc {counts[1]:,} dòng | "
        f"graph_peer {counts[2]:,} dòng",
    )
    con.close()
    log("s3", "xong.")


if __name__ == "__main__":
    main()
