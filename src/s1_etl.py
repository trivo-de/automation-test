"""
Bước 1 — ETL
Đọc 7 file CSV thô -> làm sạch -> chuẩn hoá đơn vị -> ghi vào DuckDB.

Xử lý sẵn 3 cái bẫy của bộ dữ liệu:
  1. BOM UTF-8 ở đầu 3 file  -> DuckDB tự bỏ, ta chỉ đặt lại tên cột
  2. Đơn vị giá khác nhau    -> quy hết về VND (bảng giá x 1000)
  3. Giá lệnh ngoài biên độ  -> gắn cờ is_outlier, KHÔNG xoá
"""

from config import DATA_DIR, HORIZON, connect, log


def main():
    con = connect()
    D = DATA_DIR

    log("s1", "đọc CSV thô...")

    # ---------- Danh mục mã ----------
    con.execute(f"""
    CREATE OR REPLACE TABLE sec AS
    SELECT stock_code, company_name, exchange, instrument_type,
           CAST(icb_code AS VARCHAR) AS icb_code, icb_name
    FROM read_csv_auto('{D}/securities_master.csv', header=true);
    """)

    # ---------- Giá cổ phiếu: nghìn VND -> VND ----------
    con.execute(f"""
    CREATE OR REPLACE TABLE px AS
    SELECT CAST(date AS DATE) AS d, stock_code,
           open*1000 AS o, high*1000 AS h, low*1000 AS l, close*1000 AS c,
           CAST(volume AS BIGINT) AS v
    FROM read_csv_auto('{D}/stock_market_daily.csv', header=true)
    WHERE close > 0;
    """)

    # ---------- Chỉ số ----------
    con.execute(f"""
    CREATE OR REPLACE TABLE idx AS
    SELECT CAST(date AS DATE) AS d, index_code, close AS c
    FROM read_csv_auto('{D}/market_index_daily.csv', header=true);
    """)

    # ---------- Lịch phiên giao dịch ----------
    con.execute("""
    CREATE OR REPLACE TABLE daycal AS
    SELECT d, ROW_NUMBER() OVER (ORDER BY d) AS dn
    FROM (SELECT DISTINCT d FROM px);
    """)

    # ---------- Giao dịch khách hàng (giá đã là VND) ----------
    con.execute(f"""
    CREATE OR REPLACE TABLE txn AS
    SELECT t.transaction_id, t.customer_id, t.stock_code, t.side,
           CAST(t.quantity AS BIGINT) AS qty,
           CAST(t.price AS DOUBLE)    AS price,
           CAST(t.trade_date AS DATE) AS d,
           CAST(t.execution_time AS TIME) AS ts,
           t.exchange,
           CAST(t.quantity AS BIGINT) * CAST(t.price AS DOUBLE) AS value,
           -- cờ bất thường: giá lệnh nằm ngoài biên độ cao/thấp của phiên
           CASE WHEN p.h IS NULL THEN NULL
                WHEN t.price > p.h OR t.price < p.l THEN TRUE ELSE FALSE END AS is_outlier
    FROM read_csv_auto('{D}/customer_transactions_raw.csv', header=true) t
    LEFT JOIN px p ON p.stock_code = t.stock_code AND p.d = CAST(t.trade_date AS DATE);
    """)

    # ---------- Hồ sơ khách: CHỈ giữ cột an toàn ----------
    # portfolio_value & cash_balance là giá trị CUỐI KỲ (30/12/2022) -> rò rỉ tương lai -> loại bỏ.
    con.execute(f"""
    CREATE OR REPLACE TABLE prof AS
    SELECT customer_id, customer_type, risk_level, investment_horizon,
           CAST(account_open_date AS DATE) AS open_date
    FROM read_csv_auto('{D}/customer_profile.csv', header=true);
    """)

    # ---------- Snapshot nắm giữ (dùng làm mốc ra quyết định) ----------
    con.execute(f"""
    CREATE OR REPLACE TABLE hold AS
    SELECT CAST(snapshot_date AS DATE) AS t, customer_id, stock_code,
           CAST(quantity AS BIGINT) AS qty,
           CAST(avg_cost AS DOUBLE)  AS avg_cost,
           CAST(market_value AS DOUBLE) AS mv
    FROM read_csv_auto('{D}/customer_holdings.csv', header=true);
    """)

    # ---------- Bảng khuyến nghị ----------
    con.execute(f"""
    CREATE OR REPLACE TABLE research AS
    SELECT CAST(research_date AS DATE) AS d, UPPER(TRIM(stock_code)) AS stock_code,
           CAST(market_score AS DOUBLE) AS market_score,
           UPPER(TRIM(recommendation)) AS recommendation,
           CAST(candidate_rank AS INT) AS candidate_rank
    FROM read_csv_auto('{D}/candidate_stocks_research.csv', header=true);
    """)

    # ---------- Mốc ra quyết định = 48 ngày chốt cuối tháng ----------
    con.execute("""
    CREATE OR REPLACE TABLE decision_dates AS
    SELECT h.t, k.dn
    FROM (SELECT DISTINCT t FROM hold) h
    JOIN daycal k ON k.d = h.t
    ORDER BY h.t;
    """)

    log("s1", "tính đặc trưng giá theo phiên (đà giá, biến động, thanh khoản)...")

    # ---------- Đặc trưng giá + kết quả lợi suất tương lai để audit ----------
    # y_stock/excess_fwd KHÔNG đi vào model hành vi. Chúng chỉ được dùng để đo
    # kết quả và tính tay nghề lịch sử khi cửa sổ 20 phiên đã hoàn tất trước t.
    # Dùng RANGE theo số phiên thị trường (dn) nên không bị lệch khi mã nghỉ giao dịch.
    con.execute(f"""
    CREATE OR REPLACE TABLE sd AS
    WITH b AS (
        SELECT p.stock_code, p.d, k.dn, p.c, p.v
        FROM px p JOIN daycal k ON k.d = p.d
    ),
    r AS (
        SELECT *, c / NULLIF(LAG(c) OVER (PARTITION BY stock_code ORDER BY dn), 0) - 1 AS r1
        FROM b
    ),
    w AS (
        SELECT *,
          FIRST_VALUE(c) OVER (PARTITION BY stock_code ORDER BY dn
                               RANGE BETWEEN 5  PRECEDING AND CURRENT ROW) AS c_5a,
          FIRST_VALUE(c) OVER (PARTITION BY stock_code ORDER BY dn
                               RANGE BETWEEN 20 PRECEDING AND CURRENT ROW) AS c_20a,
          FIRST_VALUE(c) OVER (PARTITION BY stock_code ORDER BY dn
                               RANGE BETWEEN 60 PRECEDING AND CURRENT ROW) AS c_60a,
          LAST_VALUE(c)  OVER (PARTITION BY stock_code ORDER BY dn
                               RANGE BETWEEN CURRENT ROW AND {HORIZON} FOLLOWING) AS c_fwd,
          STDDEV_SAMP(r1) OVER (PARTITION BY stock_code ORDER BY dn
                               RANGE BETWEEN 20 PRECEDING AND CURRENT ROW) AS vol_20,
          AVG(v) OVER (PARTITION BY stock_code ORDER BY dn
                       RANGE BETWEEN 20 PRECEDING AND CURRENT ROW) AS v_avg20,
          AVG(CASE WHEN r1 > 0 THEN 1.0 ELSE 0.0 END) OVER (PARTITION BY stock_code ORDER BY dn
                       RANGE BETWEEN 20 PRECEDING AND CURRENT ROW) AS up_ratio_20,
          COUNT(*) OVER (PARTITION BY stock_code ORDER BY dn
                         RANGE BETWEEN 20 PRECEDING AND CURRENT ROW) AS n_sessions_20
        FROM r
    ),
    vni AS (
        SELECT k.dn, i.c,
               LAST_VALUE(i.c) OVER (ORDER BY k.dn
                       RANGE BETWEEN CURRENT ROW AND {HORIZON} FOLLOWING) AS c_fwd
        FROM idx i JOIN daycal k ON k.d = i.d
        WHERE i.index_code = 'VNINDEX'
    )
    SELECT w.stock_code, w.d, w.dn, w.c, w.v,
           w.c / NULLIF(w.c_5a , 0) - 1 AS ret_5,
           w.c / NULLIF(w.c_20a, 0) - 1 AS ret_20,
           w.c / NULLIF(w.c_60a, 0) - 1 AS ret_60,
           w.vol_20, w.up_ratio_20, w.n_sessions_20,
           w.v / NULLIF(w.v_avg20, 0) AS vol_ratio,
           w.v * w.c                   AS turnover,
           w.c_fwd / NULLIF(w.c, 0) - 1                       AS fwd_ret,
           vni.c_fwd / NULLIF(vni.c, 0) - 1                    AS idx_fwd_ret,
           (w.c_fwd / NULLIF(w.c, 0)) - (vni.c_fwd / NULLIF(vni.c, 0)) AS excess_fwd,
            -- kết quả audit: mã này có thắng VNINDEX trong {HORIZON} phiên tới không?
           CASE WHEN (w.c_fwd / NULLIF(w.c, 0)) > (vni.c_fwd / NULLIF(vni.c, 0))
                THEN 1 ELSE 0 END AS y_stock
    FROM w JOIN vni ON vni.dn = w.dn;
    """)
    con.execute("CREATE INDEX IF NOT EXISTS sd_idx ON sd(stock_code, dn);")

    for t in [
        "sec",
        "px",
        "idx",
        "txn",
        "prof",
        "hold",
        "research",
        "sd",
        "decision_dates",
    ]:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        log("s1", f"  {t:16s} {n:>9,} dòng")

    bad = con.execute(
        "SELECT SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) FROM txn"
    ).fetchone()[0]
    log("s1", f"  lệnh có giá ngoài biên độ (đã gắn cờ): {bad:,}")
    con.close()
    log("s1", "xong.")


if __name__ == "__main__":
    main()
