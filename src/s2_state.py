"""
Bước 2 — Trạng thái khách hàng POINT-IN-TIME
Tại mỗi mốc quyết định t (phiên cuối tuần hoặc cuối tháng), tính lại "khách này
đang như thế nào" CHỈ bằng dữ liệu có trước hoặc bằng t.

Vị thế đang nắm được DỰNG LẠI từ giao dịch (không phụ thuộc file snapshot cuối
tháng), nên pipeline chạy được ở mọi nhịp quyết định. File snapshot chỉ còn dùng
để đối chiếu.

Kỳ hạn đầu tư (SHORT / LEAN_SHORT / MEDIUM / LEAN_LONG / LONG) cũng được suy ra ở
đây từ thời gian nắm giữ thực tế, thay cho cột investment_horizon tự khai đã không
còn trong hồ sơ.
"""

import numpy as np
import pandas as pd

from config import (
    BEHAVIOR_HALF_LIFE,
    BUY_FEE,
    HOLD_EVIDENCE_THRESHOLD,
    HOLD_MIN_EVIDENCE,
    HORIZON,
    LONG_HOLD_DAYS,
    RECENT_WINDOW,
    SHORT_HOLD_DAYS,
    connect,
    log,
)


def position_states(txn: pd.DataFrame, buy_fee: float = BUY_FEE) -> pd.DataFrame:
    """Số lượng và giá vốn bình quân sau từng ngày có giao dịch của khách x mã.

    `txn` phải được sắp theo (customer_id, stock_code, thời gian). Giá vốn theo
    phương pháp bình quân gia quyền: lệnh MUA kéo giá vốn về phía giá mua (đã
    cộng phí), lệnh BÁN không đổi giá vốn, bán hết thì giá vốn về 0.
    Trạng thái tại ngày d chỉ phụ thuộc các lệnh tới hết ngày d -> không rò rỉ.
    """
    cust = txn["customer_id"].to_numpy()
    stock = txn["stock_code"].to_numpy()
    is_buy = (txn["side"] == "BUY").to_numpy()
    qty = txn["qty"].to_numpy(dtype=np.int64)
    price = txn["price"].to_numpy(dtype=float)

    pos = np.zeros(len(txn), dtype=np.int64)
    cost = np.zeros(len(txn))
    cur_qty, cur_cost, prev = 0, 0.0, None
    for i in range(len(txn)):
        key = (cust[i], stock[i])
        if key != prev:
            cur_qty, cur_cost, prev = 0, 0.0, key
        if is_buy[i]:
            paid = qty[i] * price[i] * (1.0 + buy_fee)
            cur_cost = (cur_qty * cur_cost + paid) / (cur_qty + qty[i])
            cur_qty += qty[i]
        else:
            cur_qty -= qty[i]
            if cur_qty <= 0:  # bán hết (hoặc bán vị thế có từ trước dữ liệu)
                cur_qty, cur_cost = 0, 0.0
        pos[i], cost[i] = cur_qty, cur_cost

    out = txn[["customer_id", "stock_code", "d"]].copy()
    out["qty"], out["avg_cost"] = pos, cost
    # nhiều lệnh trong một ngày -> chỉ giữ trạng thái cuối ngày
    return out.groupby(["customer_id", "stock_code", "d"], sort=False).tail(1)


def main():
    con = connect()

    # ---------- 0. Vòng nắm giữ: mua lần đầu -> bán hết ----------
    # Một vòng bắt đầu khi khách MUA lúc đang không nắm mã đó và kết thúc khi số
    # lượng về 0. Bảng chỉ chứa NGÀY của các sự kiện, nên mọi mốc t phía sau tự
    # lọc được phần đã biết tại t (open_d < t, close_d < t...) mà không rò rỉ.
    log("s2", "tách các vòng mua -> bán hết của từng khách x mã...")
    con.execute("""
    CREATE OR REPLACE TABLE hold_episode AS
    WITH o AS (
        SELECT customer_id, stock_code, d, ts, transaction_id, side, qty, value,
               SUM(CASE WHEN side = 'BUY' THEN qty ELSE -qty END) OVER w AS pos_after
        FROM txn
        WINDOW w AS (PARTITION BY customer_id, stock_code
                     ORDER BY d, ts, transaction_id)
    ),
    e AS (
        SELECT *,
               SUM(CASE WHEN side = 'BUY'
                         AND pos_after - qty <= 0 THEN 1 ELSE 0 END) OVER w AS ep
        FROM o
        WINDOW w AS (PARTITION BY customer_id, stock_code
                     ORDER BY d, ts, transaction_id)
    )
    SELECT customer_id, stock_code, ep,
           MIN(d) AS open_d,
           MAX(CASE WHEN pos_after <= 0 THEN d END) AS close_d,
           MIN(CASE WHEN side = 'SELL' THEN d END)  AS first_sell_d,
           SUM(CASE WHEN side = 'BUY'  THEN value ELSE 0 END) AS buy_value,
           SUM(CASE WHEN side = 'SELL' THEN value ELSE 0 END) AS sell_value
    FROM e
    WHERE ep > 0          -- bỏ lệnh bán của vị thế có từ trước khi dữ liệu bắt đầu
    GROUP BY 1, 2, 3;
    """)

    # hold_episode cộng dồn số lượng mà không chặn ở 0. Nếu dữ liệu có lệnh bán một
    # vị thế có từ trước kỳ dữ liệu, số dư âm sẽ làm các vòng nắm giữ bị tách sai.
    n_negative = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT SUM(CASE WHEN side = 'BUY' THEN qty ELSE -qty END) OVER (
                       PARTITION BY customer_id, stock_code
                       ORDER BY d, ts, transaction_id) AS pos_after
            FROM txn)
        WHERE pos_after < 0
    """).fetchone()[0]
    if n_negative:
        log(
            "s2",
            f"  CẢNH BÁO: {n_negative:,} lệnh làm số dư âm (bán vị thế có từ trước kỳ "
            "dữ liệu?) — vòng nắm giữ và kỳ hạn đầu tư của các khách này có thể sai.",
        )

    # ---------- 1. Vị thế đang nắm tại t ----------
    # Dựng lại từ giao dịch: trạng thái cuối mỗi ngày có lệnh, có hiệu lực tới
    # ngày có lệnh kế tiếp. Giá trị thị trường lấy giá đóng cửa gần nhất <= t.
    log("s2", "dựng lại vị thế từ giao dịch tại mỗi mốc...")
    states = position_states(
        con.execute("""
        SELECT customer_id, stock_code, d, side, qty, price FROM txn
        ORDER BY customer_id, stock_code, d, ts, transaction_id""").df()
    )
    con.register("_pos_state_df", states)
    con.execute("""
    CREATE OR REPLACE TABLE pos_state AS
    SELECT customer_id, stock_code, CAST(d AS DATE) AS d,
           CAST(qty AS BIGINT) AS qty, CAST(avg_cost AS DOUBLE) AS avg_cost,
           LEAD(CAST(d AS DATE)) OVER (PARTITION BY customer_id, stock_code
                                       ORDER BY d) AS next_d
    FROM _pos_state_df;
    """)
    con.unregister("_pos_state_df")
    # Ba phép ghép dưới đây đều là "lấy bản ghi gần nhất <= t" nên dùng ASOF JOIN;
    # viết bằng điều kiện khoảng (d <= t AND next_d > t) chậm hơn cả trăm lần.
    con.execute("""
    CREATE OR REPLACE TABLE pos AS
    WITH p AS (
        SELECT dd.t, dd.dn, s.customer_id, s.stock_code, s.qty, s.avg_cost
        FROM decision_dates dd
        JOIN pos_state s ON s.d <= dd.t AND (s.next_d IS NULL OR s.next_d > dd.t)
        WHERE s.qty > 0
    ),
    v AS (
        SELECT p.*, p.qty * lp.c AS mv
        FROM p ASOF JOIN px lp ON lp.stock_code = p.stock_code AND p.t >= lp.d
    ),
    -- Vòng nắm giữ đang mở tại t là vòng mở gần nhất <= t. Bán hết rồi mua lại
    -- trong cùng một ngày tạo hai vòng cùng ngày mở -> lấy vòng sau.
    ep AS (
        SELECT customer_id, stock_code, open_d,
               ARG_MAX(first_sell_d, ep) AS first_sell_d
        FROM hold_episode GROUP BY 1, 2, 3
    )
    SELECT v.t, v.dn, v.customer_id, v.stock_code, v.qty, v.avg_cost, v.mv,
           v.mv / NULLIF(v.qty * v.avg_cost, 0) - 1 AS unreal_pnl_pct,
           -- vị thế này đã được giữ bao lâu và đã từng bán bớt chưa
           DATE_DIFF('day', e.open_d, v.t) AS pos_hold_days,
           CASE WHEN e.first_sell_d <= v.t THEN 1 ELSE 0 END AS pos_has_sold
    FROM v
    ASOF LEFT JOIN ep e
           ON e.customer_id = v.customer_id AND e.stock_code = v.stock_code
          AND v.t >= e.open_d;
    """)

    # Đối chiếu với snapshot cuối tháng của file gốc (so trên trạng thái dựng lại,
    # nên chạy được cả khi ngày chốt tháng không phải là một mốc quyết định).
    n_snap, n_found, n_same = con.execute("""
        SELECT COUNT(*), COUNT_IF(s.qty > 0),
               COUNT_IF(s.qty = h.qty
                        AND ABS(s.avg_cost - h.avg_cost) <= 1e-4 * h.avg_cost)
        FROM hold h
        ASOF LEFT JOIN pos_state s
               ON s.customer_id = h.customer_id AND s.stock_code = h.stock_code
              AND h.t >= s.d
    """).fetchone()
    if n_snap:
        log(
            "s2",
            f"  đối chiếu snapshot: {n_found:,}/{n_snap:,} vị thế tìm thấy, "
            f"{n_same:,} khớp cả số lượng và giá vốn "
            f"({100.0 * n_same / n_snap:.2f}%)",
        )
        if n_same < 0.99 * n_snap:
            log(
                "s2",
                "  CẢNH BÁO: vị thế dựng lại lệch snapshot quá 1% — kiểm tra "
                "BUY_FEE và dữ liệu giao dịch.",
            )

    # ---------- 2. Giá trị danh mục tại t (point-in-time) ----------
    con.execute("""
    CREATE OR REPLACE TABLE port_pit AS
    SELECT t, customer_id,
           COUNT(*)      AS n_positions,
           SUM(mv)       AS port_value,
           AVG(unreal_pnl_pct) AS avg_pnl_pct,
           -- độ tập trung: 1 = dồn hết vào một mã, gần 0 = dàn trải
           SUM(POWER(mv / NULLIF(port_total, 0), 2)) AS port_hhi
    FROM (SELECT *, SUM(mv) OVER (PARTITION BY t, customer_id) AS port_total FROM pos)
    GROUP BY 1, 2;
    """)

    # ---------- 3. Lịch sử giao dịch tích luỹ tới t ----------
    log("s2", "lịch sử giao dịch tích luỹ tới mỗi mốc...")
    # Gắn cờ lệnh mua rơi vào lúc PHS đang mở khuyến nghị cho đúng mã đó.
    con.execute("""
    CREATE OR REPLACE TABLE _txn_phs AS
    SELECT x.customer_id, x.stock_code, x.side, x.value, x.d,
           EXISTS (SELECT 1 FROM phs_calls pc
                   WHERE pc.stock_code = x.stock_code AND pc.open_d <= x.d
                     AND (pc.close_d IS NULL OR pc.close_d >= x.d)) AS in_phs
    FROM txn x;
    """)
    con.execute(f"""
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
           DATE_DIFF('day', MAX(x.d), dd.t)                    AS days_since_last_trade,
           -- nhịp giao dịch gần đây: khách đang hoạt động hay đã nguội
           SUM(CASE WHEN x.side = 'BUY'  AND x.d >= dd.t - {RECENT_WINDOW}
                    THEN 1 ELSE 0 END)                         AS n_buys_recent,
           SUM(CASE WHEN x.side = 'SELL' AND x.d >= dd.t - {RECENT_WINDOW}
                    THEN 1 ELSE 0 END)                         AS n_sells_recent,
           COUNT(DISTINCT CASE WHEN x.d >= dd.t - {RECENT_WINDOW}
                               THEN x.d END)                   AS n_active_days_recent,
           -- khách có thói quen mua theo khuyến nghị PHS không?
           SUM(CASE WHEN x.side = 'BUY' AND x.in_phs THEN 1 ELSE 0 END) AS n_phs_buys,
           SUM(CASE WHEN x.side = 'BUY' AND x.in_phs THEN 1 ELSE 0 END) * 1.0 /
               NULLIF(SUM(CASE WHEN x.side = 'BUY' THEN 1 ELSE 0 END), 0)
                                                               AS phs_follow_rate
    FROM decision_dates dd
    JOIN _txn_phs x ON x.d < dd.t
    GROUP BY 1, 2;
    """)
    # ---------- 3a. Ch?t l??ng mua theo PHS ----------
    # `phs_follow_rate` ? tr?n ch? tr? l?i "kh?ch c? hay mua theo PHS kh?ng?". B?ng n?y
    # tr? l?i th?m "nh?ng l?n mua theo PHS ?? ?? HORIZON tr??c t th? k?t qu? ra sao?".
    # ?i?u ki?n s.dn + HORIZON <= dd.dn ch?n r? r?: t?i t ch? d?ng c?c l?nh ?? bi?t
    # k?t qu? 20 phi?n.
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
    JOIN _txn_phs x ON x.side = 'BUY' AND x.in_phs AND x.d < dd.t
    JOIN sd s ON s.stock_code = x.stock_code AND s.d = x.d
    WHERE s.dn + {HORIZON} <= dd.dn
    GROUP BY 1, 2;
    """)
    con.execute("DROP TABLE IF EXISTS _txn_phs;")

    # ---------- 3b. Kỳ hạn đầu tư suy ra từ thời gian nắm giữ ----------
    # Mỗi vòng nắm giữ đã bắt đầu trước t là một bằng chứng:
    #   bán hết trong <= SHORT_HOLD_DAYS (1 tháng)            -> NGẮN HẠN
    #   giữ > LONG_HOLD_DAYS (3 tháng) rồi mới bán hết, HOẶC vẫn
    #   đang giữ > LONG_HOLD_DAYS và chưa bán bớt lần nào     -> DÀI HẠN
    #   còn lại (giữ 1-3 tháng, hoặc vị thế còn quá mới)      -> chưa kết luận
    # Mẫu số là TẤT CẢ bằng chứng, nên vòng "chưa kết luận" kéo cả hai tỷ lệ xuống.
    #
    # Tầng 1 (rõ ràng): tỷ lệ ngắn hạn / dài hạn đạt ngưỡng -> SHORT / LONG.
    # Tầng 2 (thiên về): phần lớn khách còn lại có rất nhiều bằng chứng nhưng pha
    #   trộn. Họ được xếp theo nhóm bằng chứng LỚN NHẤT trong ba nhóm đã kết luận
    #   được (ngắn / giữ 1-3 tháng / dài): LEAN_SHORT, MEDIUM, LEAN_LONG.
    # UNRATED chỉ còn dành cho khách thật sự thiếu bằng chứng.
    log("s2", "kỳ hạn đầu tư & thói quen chốt lời/cắt lỗ của khách...")
    con.execute(f"""
    CREATE OR REPLACE TABLE cust_horizon AS
    WITH ev AS (
        SELECT dd.t, e.customer_id,
               e.close_d IS NOT NULL AND e.close_d < dd.t AS closed,
               DATE_DIFF('day', e.open_d, e.close_d)       AS held_days,
               DATE_DIFF('day', e.open_d, dd.t)            AS age_days,
               e.first_sell_d IS NOT NULL AND e.first_sell_d < dd.t AS has_sold,
               GREATEST(LEAST(e.sell_value / NULLIF(e.buy_value, 0) - 1, 1.0), -1.0)
                   AS realized_ret
        FROM decision_dates dd
        JOIN hold_episode e ON e.open_d < dd.t
    ),
    k AS (
        SELECT *,
               closed AND held_days <= {SHORT_HOLD_DAYS} AS is_short,
               closed AND held_days > {SHORT_HOLD_DAYS}
                      AND held_days <= {LONG_HOLD_DAYS} AS is_mid,
               (closed AND held_days > {LONG_HOLD_DAYS})
                   OR (NOT closed AND age_days > {LONG_HOLD_DAYS} AND NOT has_sold)
                   AS is_long
        FROM ev
    ),
    a AS (
        SELECT t, customer_id,
               COUNT(*)            AS n_evidence,
               COUNT_IF(is_short)  AS n_short,
               COUNT_IF(is_long)   AS n_long,
               COUNT_IF(is_mid)    AS n_mid,
               COUNT_IF(closed)    AS n_closed,
               MEDIAN(held_days) FILTER (WHERE closed)          AS median_hold_days,
               AVG(CASE WHEN realized_ret > 0 THEN 1.0 ELSE 0.0 END)
                   FILTER (WHERE closed)                         AS win_rate,
               AVG(realized_ret) FILTER (WHERE closed)          AS avg_realized_ret,
               -- mức lãi khách thường chốt và mức lỗ khách thường chịu trước khi cắt
               AVG(realized_ret) FILTER (WHERE closed AND realized_ret > 0)
                                                                 AS avg_win_ret,
               AVG(realized_ret) FILTER (WHERE closed AND realized_ret <= 0)
                                                                 AS avg_loss_ret
        FROM k GROUP BY 1, 2
    )
    SELECT *,
           n_short * 1.0 / n_evidence AS short_ratio,
           n_mid   * 1.0 / n_evidence AS mid_ratio,
           n_long  * 1.0 / n_evidence AS long_ratio,
           CASE -- tầng 1: đạt ngưỡng trên toàn bộ bằng chứng
                WHEN n_evidence >= {HOLD_MIN_EVIDENCE}
                 AND n_short * 1.0 / n_evidence >= {HOLD_EVIDENCE_THRESHOLD}
                 AND n_long  * 1.0 / n_evidence <  {HOLD_EVIDENCE_THRESHOLD} THEN 'SHORT'
                WHEN n_evidence >= {HOLD_MIN_EVIDENCE}
                 AND n_long  * 1.0 / n_evidence >= {HOLD_EVIDENCE_THRESHOLD}
                 AND n_short * 1.0 / n_evidence <  {HOLD_EVIDENCE_THRESHOLD} THEN 'LONG'
                -- tầng 2: nhóm lớn nhất trong các vòng đã kết luận được
                WHEN n_short + n_mid + n_long < {HOLD_MIN_EVIDENCE} THEN 'UNRATED'
                WHEN n_short > n_mid AND n_short > n_long THEN 'LEAN_SHORT'
                WHEN n_long  > n_mid AND n_long  > n_short THEN 'LEAN_LONG'
                ELSE 'MEDIUM' END AS investment_horizon
    FROM a;
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


    # Kh?u v? ng?nh ??ng: ng?nh kh?ch mua g?n ??y c? ?ang l?ch kh?i kh?u v?
    # to?n l?ch s? kh?ng. C?c paper sequence/multi-view th??ng nh?n m?nh t?n hi?u n?y.
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

    # Chu?i mua g?n nh?t: m?/ng?nh ?ang x?t c? n?m trong 1/3/5 l?nh BUY g?n nh?t
    # c?a kh?ch kh?ng. T?ch stock v? sector ?? S4 gh?p v?i c? r? PHS v? danh m?c.
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
        "hold_episode",
        "pos_state",
        "pos",
        "port_pit",
        "cust_pit",
        "cust_phs_quality",
        "cust_horizon",
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
    # Phân bố kỳ hạn tại mốc mới nhất, để kiểm tra nhanh ngưỡng có hợp lý không.
    dist = con.execute("""
        SELECT investment_horizon, COUNT(*) FROM cust_horizon
        WHERE t = (SELECT MAX(t) FROM cust_horizon) GROUP BY 1 ORDER BY 1
    """).fetchall()
    log("s2", "  kỳ hạn tại mốc cuối: " + " | ".join(f"{k} {n:,}" for k, n in dist))
    con.close()
    log("s2", "xong.")


if __name__ == "__main__":
    main()
