"""Kiểm tra cột và kiểu dữ liệu của các file CSV đầu vào trước khi chạy pipeline.

Mục đích: báo lỗi sớm, đúng file, đúng cột — thay vì để một cột thiếu hoặc một ô
sai định dạng làm pipeline chết giữa chừng với thông báo SQL khó hiểu.
"""

import os

import duckdb

# kind: str | num | date | date_dmy | time | range | enum
#   date     : ISO yyyy-mm-dd            date_dmy : dd/mm/yyyy (file PHS)
#   range    : một số, hoặc vùng giá "20.0 - 20.5"
# Mỗi cột: (kind, cho_phép_trống[, tập giá trị hợp lệ của enum])
SCHEMA = {
    "securities_master.csv": {
        "stock_code": ("str", False),
        "company_name": ("str", True),
        "exchange": ("str", True),
        "instrument_type": ("str", True),
        "icb_code": ("str", True),
        "icb_name": ("str", True),
    },
    "stock_market_daily.csv": {
        "date": ("date", False),
        "stock_code": ("str", False),
        "open": ("num", True),
        "high": ("num", True),
        "low": ("num", True),
        "close": ("num", False),
        "volume": ("num", True),
    },
    "market_index_daily.csv": {
        "date": ("date", False),
        "index_code": ("str", False),
        "close": ("num", False),
    },
    "customer_profile.csv": {
        "customer_id": ("str", False),
        "customer_type": ("str", True),
        "account_open_date": ("date", False),
    },
    "customer_transactions_raw.csv": {
        "transaction_id": ("str", False),
        "customer_id": ("str", False),
        "stock_code": ("str", False),
        "side": ("enum", False, ("BUY", "SELL")),
        "quantity": ("num", False),
        "price": ("num", False),
        "trade_date": ("date", False),
        "execution_time": ("time", True),
        "exchange": ("str", True),
    },
    "customer_holdings.csv": {
        "snapshot_date": ("date", False),
        "customer_id": ("str", False),
        "stock_code": ("str", False),
        "quantity": ("num", False),
        "avg_cost": ("num", False),
        "market_value": ("num", False),
    },
    "candidate_stocks_phs_skill.csv": {
        "symbol": ("str", False),
        "recommendationType": ("enum", False, ("BUY", "TAKE_PROFIT", "CUT_LOSS")),
        "recommendationDate": ("date_dmy", False),
        "recommendationPrice": ("range", False),
        "targetPrice": ("num", False),
        "cutLossPrice": ("num", True),
        "realizedProfitLoss": ("str", True),
    },
}

REQUIRED = list(SCHEMA)


def _bad_expr(col: str, spec: tuple) -> str:
    """Biểu thức SQL trả TRUE khi một ô CÓ giá trị nhưng sai định dạng."""
    kind = spec[0]
    c = f'TRIM("{col}")'
    if kind == "num":
        return f"TRY_CAST({c} AS DOUBLE) IS NULL"
    if kind == "date":
        return f"TRY_CAST({c} AS DATE) IS NULL"
    if kind == "date_dmy":
        return f"TRY_STRPTIME({c}, '%d/%m/%Y') IS NULL"
    if kind == "time":
        return f"TRY_CAST({c} AS TIME) IS NULL"
    if kind == "range":
        return (
            f"LEN(LIST_FILTER(STR_SPLIT({c}, '-'), "
            "x -> TRY_CAST(TRIM(x) AS DOUBLE) IS NULL)) > 0"
        )
    if kind == "enum":
        allowed = ", ".join(f"'{v}'" for v in spec[2])
        return f"UPPER({c}) NOT IN ({allowed})"
    return "FALSE"


def check_file(con, path: str, columns: dict) -> list[str]:
    """Trả về danh sách lỗi của một file (rỗng nếu hợp lệ)."""
    name = os.path.basename(path)
    if not os.path.exists(path):
        return [f"{name}: không tìm thấy file"]
    try:
        rel = f"read_csv('{path}', header=true, all_varchar=true)"
        have = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()}
    except duckdb.Error as e:
        return [f"{name}: không đọc được CSV ({str(e).splitlines()[0]})"]

    missing = [c for c in columns if c not in have]
    if missing:
        return [f"{name}: thiếu cột {', '.join(missing)}"]

    selects = ["COUNT(*)"]
    for col, spec in columns.items():
        blank = f'("{col}" IS NULL OR TRIM("{col}") = \'\')'
        selects.append(f"COUNT_IF({blank})")
        selects.append(f"COUNT_IF(NOT {blank} AND ({_bad_expr(col, spec)}))")
        selects.append(
            f'MIN(CASE WHEN NOT {blank} AND ({_bad_expr(col, spec)}) THEN "{col}" END)'
        )
    row = con.execute(f"SELECT {', '.join(selects)} FROM {rel}").fetchone()

    n = row[0]
    if n == 0:
        return [f"{name}: file không có dòng dữ liệu nào"]
    errors = []
    for i, (col, spec) in enumerate(columns.items()):
        n_blank, n_bad, example = row[1 + 3 * i : 4 + 3 * i]
        if n_blank and not spec[1]:
            errors.append(f"{name}: cột {col} bị trống ở {n_blank:,}/{n:,} dòng")
        if n_bad:
            errors.append(
                f"{name}: cột {col} có {n_bad:,} giá trị sai kiểu {spec[0]} "
                f"(ví dụ: {example!r})"
            )
    return errors


def validate_inputs(data_dir: str) -> list[str]:
    """Kiểm tra toàn bộ file đầu vào; trả về mọi lỗi tìm được."""
    con = duckdb.connect()
    errors = []
    try:
        for fname, columns in SCHEMA.items():
            errors += check_file(con, os.path.join(data_dir, fname), columns)
        idx = os.path.join(data_dir, "market_index_daily.csv")
        if not errors and os.path.exists(idx):
            has_vni = con.execute(
                f"SELECT COUNT(*) FROM read_csv('{idx}', header=true, "
                "all_varchar=true) WHERE UPPER(TRIM(index_code)) = 'VNINDEX'"
            ).fetchone()[0]
            if not has_vni:
                errors.append(
                    "market_index_daily.csv: không có index_code = 'VNINDEX' "
                    "(cần để tính lợi suất vượt trội)"
                )
    finally:
        con.close()
    return errors
