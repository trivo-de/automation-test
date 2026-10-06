"""
Bước 1 — ETL
Đọc 7 file CSV thô (6 file dữ liệu + nhật ký khuyến nghị PHS) -> làm sạch -> chuẩn hoá đơn vị -> ghi vào DuckDB.

Xử lý sẵn 3 cái bẫy của bộ dữ liệu:
  1. BOM UTF-8 ở đầu 3 file  -> DuckDB tự bỏ, ta chỉ đặt lại tên cột
  2. Đơn vị giá khác nhau    -> quy hết về VND (bảng giá x 1000)
  3. Giá lệnh ngoài biên độ  -> gắn cờ is_outlier, KHÔNG xoá
"""

from config import (
    DATA_DIR,
    DECISION_FREQ,
    HORIZON,
    SAMPLE_PCT,
    SAMPLE_SEED,
    connect,
    log,
)


def sample(column: str) -> str:
    """Điều kiện SQL giữ lại SAMPLE_PCT % khách. Chọn theo băm của mã khách nên một
    khách hoặc có đủ mọi giao dịch, hoặc vắng hẳn — lịch sử từng người không bị cắt."""
    if SAMPLE_PCT >= 100:
        return "TRUE"
    return f"HASH({column} || '{SAMPLE_SEED}') % 100 < {SAMPLE_PCT}"


def main():
    con = connect()
    D = DATA_DIR
    if SAMPLE_PCT < 100:
        log(
            "s1",
            f"CHẾ ĐỘ CHẠY NHANH: chỉ giữ {SAMPLE_PCT}% khách (seed {SAMPLE_SEED}).",
        )

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
    LEFT JOIN px p ON p.stock_code = t.stock_code AND p.d = CAST(t.trade_date AS DATE)
    WHERE {sample("t.customer_id")};
    """)

    # ---------- Hồ sơ khách ----------
    # File hồ sơ chỉ còn loại khách + ngày mở tài khoản. Kỳ hạn đầu tư không còn
    # được khai báo sẵn mà suy ra từ hành vi nắm giữ thực tế ở s2 (cust_horizon).
    con.execute(f"""
    CREATE OR REPLACE TABLE prof AS
    SELECT customer_id, customer_type,
           CAST(account_open_date AS DATE) AS open_date
    FROM read_csv_auto('{D}/customer_profile.csv', header=true)
    WHERE {sample("customer_id")};
    """)

    # ---------- Snapshot nắm giữ cuối tháng ----------
    # Chỉ còn dùng để ĐỐI CHIẾU: vị thế tại mọi mốc được dựng lại từ giao dịch ở
    # s2, và phải khớp với file này tại các ngày chốt tháng.
    con.execute(f"""
    CREATE OR REPLACE TABLE hold AS
    SELECT CAST(snapshot_date AS DATE) AS t, customer_id, stock_code,
           CAST(quantity AS BIGINT) AS qty,
           CAST(avg_cost AS DOUBLE)  AS avg_cost,
           CAST(market_value AS DOUBLE) AS mv
    FROM read_csv_auto('{D}/customer_holdings.csv', header=true)
    WHERE {sample("customer_id")};
    """)

    # ---------- Mốc ra quyết định = từng phiên giao dịch / phiên cuối tuần / cuối tháng ----------
    if DECISION_FREQ == "day":
        con.execute("""
        CREATE OR REPLACE TABLE decision_dates AS
        SELECT d AS t, dn
        FROM daycal
        ORDER BY d;
        """)
    else:
        con.execute(f"""
    CREATE OR REPLACE TABLE decision_dates AS
    SELECT d AS t, dn
    FROM daycal
    QUALIFY d = MAX(d) OVER (PARTITION BY DATE_TRUNC('{DECISION_FREQ}', d))
    ORDER BY d;
    """)

    # ---------- Khuyến nghị PHS: ghép lệnh MỞ với lệnh ĐÓNG ----------
    # File gốc là nhật ký sự kiện: BUY mở một khuyến nghị, TAKE_PROFIT/CUT_LOSS
    # đóng nó. Lệnh BUY thứ k của một mã được đóng bởi lệnh đóng thứ k của mã đó.
    # Giá trong file tính theo nghìn VND, vùng giá "20.0 - 20.5" lấy điểm giữa.
    con.execute(f"""
    CREATE OR REPLACE TABLE phs_calls AS
    WITH ev AS (
        SELECT UPPER(TRIM(symbol)) AS stock_code,
               UPPER(TRIM(recommendationType)) AS call_type,
               CAST(STRPTIME(TRIM(recommendationDate), '%d/%m/%Y') AS DATE) AS d,
               LIST_AVG(LIST_TRANSFORM(STR_SPLIT(recommendationPrice, '-'),
                        x -> CAST(TRIM(x) AS DOUBLE))) * 1000 AS price,
               CAST(targetPrice AS DOUBLE) * 1000  AS target_price,
               CAST(cutLossPrice AS DOUBLE) * 1000 AS cut_loss_price,
               CAST(REPLACE(NULLIF(TRIM(realizedProfitLoss), ''), '%', '') AS DOUBLE)
                   AS realized_pnl_pct
        FROM read_csv('{D}/candidate_stocks_phs_skill.csv', header=true, all_varchar=true)
    ),
    o AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY d, target_price) AS k
        FROM ev WHERE call_type = 'BUY'
    ),
    c AS (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY d, target_price) AS k
        FROM ev WHERE call_type IN ('TAKE_PROFIT', 'CUT_LOSS')
    )
    SELECT o.stock_code, o.d AS open_d, c.d AS close_d, c.call_type AS close_type,
           o.price AS entry_price, o.target_price, o.cut_loss_price,
           c.realized_pnl_pct
    FROM o LEFT JOIN c ON c.stock_code = o.stock_code AND c.k = o.k AND c.d >= o.d;
    """)

    # ---------- Tín hiệu nghiên cứu tại từng mốc t (point-in-time) ----------
    #   BUY  : PHS mở khuyến nghị mới đúng ngày t.
    #   SELL : PHS chốt lời/cắt lỗ đúng ngày t.
    #   HOLD : khuyến nghị BUY cũ còn mở, chưa có tín hiệu mới trong ngày.
    # Không lặp lại BUY cũ thành khuyến nghị mua mới ở các ngày sau.
    con.execute(f"""
    CREATE OR REPLACE TABLE research AS
    WITH buy AS (
        SELECT dd.t AS d, pc.stock_code, 'BUY' AS recommendation, 'BUY' AS call_type,
               pc.open_d AS call_date, pc.entry_price, pc.target_price, pc.cut_loss_price,
               100.0 * (pc.target_price / NULLIF(COALESCE(p.c, pc.entry_price), 0) - 1)
                   AS market_score
        FROM decision_dates dd
        JOIN phs_calls pc ON pc.open_d = dd.t
        LEFT JOIN px p ON p.stock_code = pc.stock_code AND p.d = dd.t
        QUALIFY ROW_NUMBER() OVER (PARTITION BY dd.t, pc.stock_code
                                   ORDER BY pc.open_d DESC) = 1
    ),
    sell AS (
        SELECT dd.t AS d, pc.stock_code, 'SELL' AS recommendation,
               pc.close_type AS call_type, pc.close_d AS call_date,
               pc.entry_price, pc.target_price, pc.cut_loss_price,
               CAST(NULL AS DOUBLE) AS market_score
        FROM decision_dates dd
        JOIN phs_calls pc ON pc.close_d = dd.t
        QUALIFY ROW_NUMBER() OVER (PARTITION BY dd.t, pc.stock_code
                                   ORDER BY pc.close_d DESC) = 1
    ),
    hold AS (
        SELECT dd.t AS d, pc.stock_code, 'HOLD' AS recommendation,
               'HOLD' AS call_type, pc.open_d AS call_date,
               pc.entry_price, pc.target_price, pc.cut_loss_price,
               CAST(NULL AS DOUBLE) AS market_score
        FROM decision_dates dd
        JOIN phs_calls pc ON pc.open_d < dd.t
                         AND (pc.close_d IS NULL OR pc.close_d > dd.t)
        WHERE NOT EXISTS (SELECT 1 FROM buy b
                          WHERE b.d = dd.t AND b.stock_code = pc.stock_code)
          AND NOT EXISTS (SELECT 1 FROM sell s
                          WHERE s.d = dd.t AND s.stock_code = pc.stock_code)
        QUALIFY ROW_NUMBER() OVER (PARTITION BY dd.t, pc.stock_code
                                   ORDER BY pc.open_d DESC) = 1
    )
    SELECT d, stock_code, market_score, recommendation,
           CAST(RANK() OVER (PARTITION BY d ORDER BY market_score DESC) AS INT)
               AS candidate_rank,
           call_type, call_date, entry_price, target_price, cut_loss_price
    FROM buy
    UNION ALL
    SELECT d, stock_code, market_score, recommendation, CAST(NULL AS INT),
           call_type, call_date, entry_price, target_price, cut_loss_price
    FROM sell
    WHERE NOT EXISTS (SELECT 1 FROM buy b
                      WHERE b.d = sell.d AND b.stock_code = sell.stock_code)
    UNION ALL
    SELECT d, stock_code, market_score, recommendation, CAST(NULL AS INT),
           call_type, call_date, entry_price, target_price, cut_loss_price
    FROM hold
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
        "phs_calls",
        "research",
        "sd",
        "decision_dates",
    ]:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        log("s1", f"  {t:16s} {n:>9,} dòng")

    n_calls, n_seen = con.execute("""
        SELECT COUNT(*), COUNT_IF(EXISTS (
            SELECT 1 FROM decision_dates dd
            WHERE dd.t >= pc.open_d AND (pc.close_d IS NULL OR dd.t < pc.close_d)))
        FROM phs_calls pc
    """).fetchone()
    log(
        "s1",
        f"  nhịp quyết định '{DECISION_FREQ}': {n_seen}/{n_calls} khuyến nghị PHS "
        "còn mở tại ít nhất một mốc",
    )

    n_dates, avg_buy, n_empty = con.execute("""
        SELECT COUNT(*), AVG(n_buy), COUNT_IF(n_buy = 0) FROM (
            SELECT dd.t, COUNT(r.stock_code) AS n_buy
            FROM decision_dates dd
            LEFT JOIN research r ON r.d = dd.t AND r.recommendation = 'BUY'
            GROUP BY 1)
    """).fetchone()
    log(
        "s1",
        f"  rổ PHS BUY mới trong ngày: TB {avg_buy:.1f} mã/mốc | "
        f"{n_empty}/{n_dates} mốc không có mã nào",
    )

    bad = con.execute(
        "SELECT SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) FROM txn"
    ).fetchone()[0]
    log("s1", f"  lệnh có giá ngoài biên độ (đã gắn cờ): {bad:,}")
    con.close()
    log("s1", "xong.")


if __name__ == "__main__":
    main()
