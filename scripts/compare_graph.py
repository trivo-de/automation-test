"""Đối chiếu đồ thị Neo4j với bản SQL: hai cách tính phải ra cùng ba bảng.

Chạy trong container app khi pipeline chính đã xong (chỉ đọc DuckDB chính):
    docker compose exec -T app python - < scripts/compare_graph.py

Tính lại graph_pop / graph_cooc / graph_peer bằng s3_graph_sql trên một DuckDB tạm rồi
so từng dòng với bảng do Neo4j tạo ra. Thoát mã 1 nếu lệch.
"""

import os
import sys
import tempfile

sys.path.insert(0, "/app/src")

import duckdb

import s3_graph_sql
from config import DUCKDB_MEMORY, DUCKDB_PATH, DUCKDB_THREADS

TABLES = {
    "graph_pop": (
        ["qt", "stock_code"],
        ["n_buyers", "buy_value", "n_sellers", "sell_value"],
    ),
    "graph_cooc": (["qt", "a", "b"], ["co"]),
    "graph_peer": (["qt", "customer_id", "peer_id"], ["sim"]),
}


def main():
    tmp = tempfile.mkdtemp(prefix="vv-graph-")
    con = duckdb.connect(os.path.join(tmp, "sql.duckdb"))
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'; SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute(f"ATTACH '{DUCKDB_PATH}' AS main_db (READ_ONLY)")
    con.execute("""
        CREATE TABLE txn AS SELECT customer_id, stock_code, d, side, value FROM main_db.txn;
        CREATE TABLE decision_dates AS SELECT * FROM main_db.decision_dates;
    """)
    s3_graph_sql.build(con)

    ok = True
    for table, (keys, values) in TABLES.items():
        on = " AND ".join(f"n.{k} = s.{k}" for k in keys)
        diff = " OR ".join(
            f"ABS(n.{v} - s.{v}) > 1e-9 * GREATEST(ABS(n.{v}), 1)" for v in values
        )
        n_neo, n_sql, only_neo, only_sql, n_diff = con.execute(f"""
            SELECT (SELECT COUNT(*) FROM main_db.{table}),
                   (SELECT COUNT(*) FROM {table}),
                   COUNT_IF(s.{keys[0]} IS NULL), COUNT_IF(n.{keys[0]} IS NULL),
                   COUNT_IF(n.{keys[0]} IS NOT NULL AND s.{keys[0]} IS NOT NULL
                            AND ({diff}))
            FROM main_db.{table} n FULL JOIN {table} s ON {on}
        """).fetchone()
        same = only_neo == 0 and only_sql == 0 and n_diff == 0
        ok &= same
        print(
            f"  [{'OK ' if same else 'LỖI'}] {table}: Neo4j {n_neo:,} dòng | SQL {n_sql:,} dòng"
            f" | chỉ có ở Neo4j {only_neo:,} | chỉ có ở SQL {only_sql:,} | lệch giá trị {n_diff:,}"
        )
    con.close()
    if not ok:
        sys.exit(1)
    print("Hai cách tính cho cùng kết quả.")


if __name__ == "__main__":
    main()
