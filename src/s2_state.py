"""
Bước 2 — Trạng thái khách hàng POINT-IN-TIME
Tại mỗi mốc t (48 ngày chốt cuối tháng), tính lại "khách này đang như thế nào"
CHỈ bằng dữ liệu có trước hoặc bằng t. Đây là thứ thay thế cho 2 cột
portfolio_value / cash_balance trong customer_profile.csv (vốn là giá trị cuối kỳ -> rò rỉ).
"""

from config import BEHAVIOR_HALF_LIFE, HORIZON, RECENT_WINDOW, connect, log


def main():
    con = connect()

    # ---------- 1. Vị thế đang nắm tại t ----------
    # Bảng hold vốn đã là snapshot cuối tháng và đã đối chiếu khớp 100% với txn,
    # nên dùng trực tiếp, chỉ gắn thêm đặc trưng giá của mã tại đúng ngày t.
    log("s2", "vị thế đang nắm tại mỗi mốc...")
    con.execute("""
    CREATE OR REPLACE TABLE pos AS
    SELECT h.t, d.dn, h.customer_id, h.stock_code, h.qty, h.avg_cost, h.mv,
           h.mv / NULLIF(h.qty * h.avg_cost, 0) - 1 AS unreal_pnl_pct
    FROM hold h JOIN decision_dates d ON d.t = h.t;
    """)

    # ---------- 2. Giá trị danh mục tại t (point-in-time) ----------
    con.execute("""
    CREATE OR REPLACE TABLE port_pit AS
    SELECT t, customer_id,
           COUNT(*)      AS n_positions,
           SUM(mv)       AS port_value,
           AVG(unreal_pnl_pct) AS avg_pnl_pct
    FROM pos GROUP BY 1, 2;
    """)

    # ---------- 3. Lịch sử giao dịch tích luỹ tới t ----------
    log("s2", "lịch sử giao dịch tích luỹ tới mỗi mốc...")
    con.execute("""
    CREATE OR REPLACE TABLE cust_pit AS
    SELECT dd.t, x.customer_id,
           COUNT(*)                                            AS n_trades,
           COUNT(DISTINCT x.stock_code)                        AS n_stocks,
           SUM(CASE WHEN x.side = 'BUY' THEN 1 ELSE 0 END)     AS n_buys,
           SUM(CASE WHEN x.side = 'SELL' THEN 1 ELSE 0 END)    AS n_sells,
           SUM(CASE WHEN x.side = 'BUY' THEN x.value ELSE 0 END)  AS buy_value,
           SUM(CASE WHEN x.side = 'SELL' THEN x.value ELSE 0 END) AS sell_value,
           AVG(x.value)                                        AS avg_trade_value,
           DATE_DIFF('day', MIN(x.d), dd.t)                    AS days_since_first_trade,
           DATE_DIFF('day', MAX(x.d), dd.t)                    AS days_since_last_trade
    FROM decision_dates dd
    JOIN txn x ON x.d < dd.t
    GROUP BY 1, 2;
    """)

    # Upstream lưu research theo ngày: chỉ tính lệnh mua có tín hiệu BUY cùng ngày.
    # Chỉ dùng kết quả đã đủ HORIZON phiên tại t; EXISTS tránh nhân đôi giao dịch.
    con.execute(f"""
    CREATE OR REPLACE TABLE cust_phs_quality AS
    SELECT dd.t, x.customer_id,
           COUNT(*) AS phs_scored_buys,
           AVG(CAST(s.y_stock AS DOUBLE)) AS phs_hit_rate,
           AVG(s.excess_fwd) AS phs_avg_excess,
           COUNT(*) FILTER (WHERE x.d >= dd.t - {RECENT_WINDOW}) AS phs_recent_scored_buys,
           AVG(CAST(s.y_stock AS DOUBLE)) FILTER (WHERE x.d >= dd.t - {RECENT_WINDOW})
               AS phs_recent_hit_rate,
           AVG(s.excess_fwd) FILTER (WHERE x.d >= dd.t - {RECENT_WINDOW})
               AS phs_recent_avg_excess
    FROM decision_dates dd
    JOIN txn x ON x.side = 'BUY' AND x.d < dd.t
    JOIN sd s ON s.stock_code = x.stock_code AND s.d = x.d
    WHERE s.dn + {HORIZON} <= dd.dn
      AND EXISTS (SELECT 1 FROM research r
                  WHERE r.stock_code = x.stock_code AND r.d = x.d
                    AND r.recommendation = 'BUY')
    GROUP BY 1, 2;
    """)

    # ---------- 4. "Tay nghề" khách trong quá khứ ----------
    # Tỷ lệ lệnh MUA trước đây mà 20 phiên sau mã đó thắng VNINDEX.
    # Chỉ tính các lệnh đã ĐỦ 20 phiên TRƯỚC t, nếu không sẽ rò rỉ tương lai.
    log("s2", "tỷ lệ chọn đúng mã trong quá khứ (đã chặn rò rỉ)...")
    con.execute(f"""
    CREATE OR REPLACE TABLE cust_skill AS
    SELECT dd.t, x.customer_id,
           AVG(CAST(s.y_stock AS DOUBLE)) AS hist_hit_rate,
           COUNT(*)                        AS n_scored_buys
    FROM decision_dates dd
    JOIN txn x ON x.side = 'BUY' AND x.d < dd.t
    JOIN sd  s ON s.stock_code = x.stock_code AND s.d = x.d
    WHERE s.dn + {HORIZON} <= dd.dn
    GROUP BY 1, 2;
    """)

    # ---------- 5. Khẩu vị ngành tới t ----------
    log("s2", "khẩu vị ngành ICB tới mỗi mốc...")
    con.execute("""
    CREATE OR REPLACE TABLE cust_sector AS
    SELECT dd.t, x.customer_id, sc.icb_code,
           COUNT(*) AS n_icb,
           COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY dd.t, x.customer_id) AS icb_share
    FROM decision_dates dd
    JOIN txn x  ON x.d < dd.t AND x.side = 'BUY'
    JOIN sec sc ON sc.stock_code = x.stock_code
    GROUP BY 1, 2, 3;
    """)

    # Khẩu vị ngành trong 30 ngày và RECENT_WINDOW ngày so với toàn bộ lịch sử.
    con.execute(f"""
    CREATE OR REPLACE TABLE cust_sector_recent AS
    WITH a AS (
        SELECT dd.t, x.customer_id, sc.icb_code,
               COUNT(*) AS n_all,
               SUM(x.value) AS value_all,
               SUM(CASE WHEN x.d >= dd.t - 30 THEN 1 ELSE 0 END) AS n_30d,
               SUM(CASE WHEN x.d >= dd.t - 30 THEN x.value ELSE 0 END) AS value_30d,
               SUM(CASE WHEN x.d >= dd.t - {RECENT_WINDOW} THEN 1 ELSE 0 END) AS n_90d,
               SUM(CASE WHEN x.d >= dd.t - {RECENT_WINDOW} THEN x.value ELSE 0 END)
                   AS value_90d
        FROM decision_dates dd
        JOIN txn x  ON x.d < dd.t AND x.side = 'BUY'
        JOIN sec sc ON sc.stock_code = x.stock_code
        GROUP BY 1, 2, 3
    )
    SELECT *,
           n_all * 1.0 / NULLIF(SUM(n_all) OVER (PARTITION BY t, customer_id), 0)
               AS icb_share_all,
           n_30d * 1.0 / NULLIF(SUM(n_30d) OVER (PARTITION BY t, customer_id), 0)
               AS icb_share_30d,
           n_90d * 1.0 / NULLIF(SUM(n_90d) OVER (PARTITION BY t, customer_id), 0)
               AS icb_share_90d,
           value_90d * 1.0 / NULLIF(SUM(value_90d) OVER (PARTITION BY t, customer_id), 0)
               AS icb_value_share_90d,
           COALESCE(
               n_30d * 1.0 / NULLIF(SUM(n_30d) OVER (PARTITION BY t, customer_id), 0), 0
           ) - n_all * 1.0 / NULLIF(SUM(n_all) OVER (PARTITION BY t, customer_id), 0)
               AS icb_recent_shift_30d,
           COALESCE(
               n_90d * 1.0 / NULLIF(SUM(n_90d) OVER (PARTITION BY t, customer_id), 0), 0
           ) - n_all * 1.0 / NULLIF(SUM(n_all) OVER (PARTITION BY t, customer_id), 0)
               AS icb_recent_shift_90d
    FROM a;
    """)

    # Xếp lệnh mua theo ngày, giờ khớp và mã giao dịch để phá hòa ổn định.
    con.execute("""
    CREATE OR REPLACE TABLE cust_recent_stock_seq AS
    WITH r AS (
        SELECT dd.t, x.customer_id, x.stock_code,
               ROW_NUMBER() OVER (
                   PARTITION BY dd.t, x.customer_id
                   ORDER BY x.d DESC, x.ts DESC, x.transaction_id DESC
               ) AS rn,
               DATE_DIFF('day', x.d, dd.t) AS days_ago
        FROM decision_dates dd
        JOIN txn x ON x.d < dd.t AND x.side = 'BUY'
    )
    SELECT t, customer_id, stock_code,
           MAX(CASE WHEN rn = 1 THEN 1 ELSE 0 END) AS stock_in_last_buy,
           MAX(CASE WHEN rn <= 3 THEN 1 ELSE 0 END) AS stock_in_last_3_buys,
           MAX(CASE WHEN rn <= 5 THEN 1 ELSE 0 END) AS stock_in_last_5_buys,
           MIN(days_ago) AS days_since_last_buy_stock
    FROM r
    GROUP BY 1, 2, 3;

    CREATE OR REPLACE TABLE cust_recent_sector_seq AS
    WITH r AS (
        SELECT dd.t, x.customer_id, sc.icb_code,
               ROW_NUMBER() OVER (
                   PARTITION BY dd.t, x.customer_id
                   ORDER BY x.d DESC, x.ts DESC, x.transaction_id DESC
               ) AS rn,
               DATE_DIFF('day', x.d, dd.t) AS days_ago
        FROM decision_dates dd
        JOIN txn x ON x.d < dd.t AND x.side = 'BUY'
        JOIN sec sc ON sc.stock_code = x.stock_code
    )
    SELECT t, customer_id, icb_code,
           MAX(CASE WHEN rn = 1 THEN 1 ELSE 0 END) AS same_sector_as_last_buy,
           SUM(CASE WHEN rn <= 3 THEN 1 ELSE 0 END) AS same_sector_last_3_buys,
           SUM(CASE WHEN rn <= 5 THEN 1 ELSE 0 END) AS same_sector_last_5_buys,
           MIN(days_ago) AS days_since_last_buy_same_sector
    FROM r
    GROUP BY 1, 2, 3;
    """)

    # ---------- 6. Phong cách đầu tư lịch sử ----------
    # Học từ trạng thái thị trường NGAY TRƯỚC các lệnh BUY cũ. Trọng số giảm một
    # nửa sau BEHAVIOR_HALF_LIFE ngày, giống ý tưởng style profile trong notebook.
    log("s2", "phong cách biến động/đà giá/thanh khoản của khách...")
    con.execute(f"""
    CREATE OR REPLACE TABLE cust_style AS
    WITH z AS (
        SELECT dd.t, x.customer_id, s.ret_20, s.vol_20,
               LN(GREATEST(s.turnover, 1)) AS log_turnover,
               x.value * POWER(0.5,
                   DATE_DIFF('day', x.d, dd.t) * 1.0 / {BEHAVIOR_HALF_LIFE}) AS w
        FROM decision_dates dd
        JOIN txn x ON x.d < dd.t AND x.side = 'BUY'
        JOIN sd s ON s.stock_code = x.stock_code AND s.d = x.d
    )
    SELECT t, customer_id,
           SUM(ret_20 * w) / NULLIF(SUM(CASE WHEN ret_20 IS NOT NULL THEN w ELSE 0 END), 0)
               AS preferred_ret_20,
           SUM(vol_20 * w) / NULLIF(SUM(CASE WHEN vol_20 IS NOT NULL THEN w ELSE 0 END), 0)
               AS preferred_vol_20,
           SUM(log_turnover * w) / NULLIF(SUM(CASE WHEN log_turnover IS NOT NULL THEN w ELSE 0 END), 0)
               AS preferred_log_turnover
    FROM z GROUP BY 1, 2;
    """)

    # ---------- 7. Quan hệ khách x mã tới t ----------
    log("s2", "lịch sử khách x mã tới mỗi mốc...")
    con.execute("""
    CREATE OR REPLACE TABLE cust_stock_pit AS
    SELECT dd.t, x.customer_id, x.stock_code,
           COUNT(*)                          AS n_trades_stock,
           DATE_DIFF('day', MAX(x.d), dd.t)  AS days_since_stock,
           SUM(CASE WHEN x.side = 'BUY' THEN 1 ELSE 0 END)  AS n_buys_stock,
           SUM(CASE WHEN x.side = 'SELL' THEN 1 ELSE 0 END) AS n_sells_stock,
           DATE_DIFF('day', MAX(CASE WHEN x.side = 'BUY' THEN x.d END), dd.t)
               AS days_since_buy,
           DATE_DIFF('day', MAX(CASE WHEN x.side = 'SELL' THEN x.d END), dd.t)
               AS days_since_sell,
           SUM(CASE WHEN x.side = 'BUY' THEN x.value ELSE 0 END)  AS buy_value,
           SUM(CASE WHEN x.side = 'SELL' THEN x.value ELSE 0 END) AS sell_value,
           ARG_MAX(x.side, x.d) AS last_side
    FROM decision_dates dd
    JOIN txn x ON x.d < dd.t
    GROUP BY 1, 2, 3;
    """)

    for t in [
        "pos",
        "port_pit",
        "cust_pit",
        "cust_phs_quality",
        "cust_skill",
        "cust_sector",
        "cust_sector_recent",
        "cust_recent_stock_seq",
        "cust_recent_sector_seq",
        "cust_style",
        "cust_stock_pit",
    ]:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        log("s2", f"  {t:16s} {n:>9,} dòng")
    con.close()
    log("s2", "xong.")


if __name__ == "__main__":
    main()
