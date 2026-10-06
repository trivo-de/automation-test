"""Cấu hình dùng chung — đọc từ biến môi trường (.env)."""

import os

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
OUT_DIR = os.getenv("OUT_DIR", "/app/artifacts")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/app/artifacts/vienvien.duckdb")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")  # giá trị thật nằm trong .env

HORIZON = int(os.getenv("HORIZON", "20"))  # số phiên nhìn tới (nhãn)
TOPK = int(os.getenv("TOPK", "10"))
# Nhịp ra quyết định: mặc định là từng phiên giao dịch. Có thể chuyển về phiên cuối
# mỗi "week" hoặc "month" để so sánh với các phiên bản cũ.
DECISION_FREQ = os.getenv("DECISION_FREQ", "day").strip().lower()
if DECISION_FREQ not in ("day", "week", "month"):
    raise ValueError("DECISION_FREQ phải là 'day', 'week' hoặc 'month'")
COOC_WINDOW = int(os.getenv("COOC_WINDOW", "180"))  # cửa sổ đồng mua (ngày)
COOC_TOPN = int(os.getenv("COOC_TOPN", "25"))  # số mã đồng mua mạnh nhất / mã

# Điểm cuối cho nhánh MUA = MARKET_WEIGHT * điểm nghiên cứu
#                              + (1-MARKET_WEIGHT) * điểm hành vi khách.
MARKET_WEIGHT = float(os.getenv("MARKET_WEIGHT", "0.60"))
BEHAVIOR_HALF_LIFE = int(os.getenv("BEHAVIOR_HALF_LIFE", "90"))

if not 0 <= MARKET_WEIGHT <= 1:
    raise ValueError("MARKET_WEIGHT phải nằm trong [0, 1]")

# Phân loại kỳ hạn đầu tư của khách từ thời gian nắm giữ thực tế (xem s2_state).
SHORT_HOLD_DAYS = int(os.getenv("SHORT_HOLD_DAYS", "30"))  # bán hết trong <= 1 tháng
LONG_HOLD_DAYS = int(os.getenv("LONG_HOLD_DAYS", "90"))  # giữ > 3 tháng
# tỷ lệ bằng chứng tối thiểu để kết luận khách SHORT hoặc LONG
HOLD_EVIDENCE_THRESHOLD = float(os.getenv("HOLD_EVIDENCE_THRESHOLD", "0.50"))
# số vòng mua-bán tối thiểu trước khi dám kết luận
HOLD_MIN_EVIDENCE = int(os.getenv("HOLD_MIN_EVIDENCE", "3"))
# cửa sổ (ngày) đo mức độ giao dịch gần đây của khách
RECENT_WINDOW = int(os.getenv("RECENT_WINDOW", "90"))

if not 0 < SHORT_HOLD_DAYS < LONG_HOLD_DAYS:
    raise ValueError("Cần 0 < SHORT_HOLD_DAYS < LONG_HOLD_DAYS")
if not 0 < HOLD_EVIDENCE_THRESHOLD <= 1:
    raise ValueError("HOLD_EVIDENCE_THRESHOLD phải nằm trong (0, 1]")

# Phí mua tính vào giá vốn (0,15%) — khớp với avg_cost trong customer_holdings.csv.
BUY_FEE = float(os.getenv("BUY_FEE", "0.0015"))

# ---------- Đồ thị: khách tương tự (Neo4j) ----------
# số khách tương tự giữ lại cho mỗi khách, và số mã chung tối thiểu để tính là giống
PEER_TOPK = int(os.getenv("PEER_TOPK", "20"))
PEER_MIN_SHARED = int(os.getenv("PEER_MIN_SHARED", "2"))

# số lát cắt thời gian gửi tới Neo4j cùng lúc ở bước S3
GRAPH_WORKERS = int(os.getenv("GRAPH_WORKERS", "2"))

# ---------- Trọng số đặt tay của các điểm thành phần ----------
# Gom về một chỗ để dễ thử nghiệm. Các điểm này chỉ là ĐẶC TRƯNG đưa vào model
# hành vi; điểm cuối vẫn do MARKET_WEIGHT ở trên quyết định.
# graph_score = cooc * đồng mua + buyer_pop * độ phổ biến
GRAPH_SCORE_WEIGHTS = {"cooc": 0.75, "buyer_pop": 0.25}
# historical_preference = tỷ trọng giá trị mua + tỷ trọng số lệnh mua + độ gần đây
HISTORY_PREF_WEIGHTS = {"value_share": 0.40, "count_share": 0.30, "recency": 0.30}
# personalization_score = graph + phù hợp phong cách + lịch sử đúng mã
PERSONALIZATION_WEIGHTS = {"graph": 0.50, "suitability": 0.30, "history": 0.20}
# độ rộng (mẫu số trong EXP(-|lệch| / scale)) khi so mã với phong cách của khách
STYLE_SCALES = {"vol_20": 0.03, "ret_20": 0.15, "log_turnover": 5.0}
# Nhánh danh mục: bậc của tín hiệu nghiên cứu và mức behavior được phép phá hoà.
# RESEARCH_TIEBREAK < 0.5 để behavior không lật được thứ tự giữa hai bậc.
RESEARCH_SELL_SIGNAL = {"SELL": 1.5, "BUY": 0.5, "HOLD": 0.0, "NONE": -0.5}
RESEARCH_TIEBREAK = 0.49

for _name, _w in (
    ("GRAPH_SCORE_WEIGHTS", GRAPH_SCORE_WEIGHTS),
    ("HISTORY_PREF_WEIGHTS", HISTORY_PREF_WEIGHTS),
    ("PERSONALIZATION_WEIGHTS", PERSONALIZATION_WEIGHTS),
):
    if abs(sum(_w.values()) - 1.0) > 1e-9:
        raise ValueError(f"{_name} phải có tổng bằng 1")

# ---------- Huấn luyện & đánh giá ----------
# Các mức K để đo xếp hạng, không phụ thuộc TOPK hiển thị trong app.
EVAL_KS = (1, 3)
# Lưới MARKET_WEIGHT được đo lại sau mỗi lần chạy (eval_buy_weights.csv).
MARKET_WEIGHT_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# ---------- Chế độ chạy nhanh & thử nghiệm ----------
# % khách hàng giữ lại (chọn cố định theo mã khách). 100 = toàn bộ. Nhỏ hơn 100 là
# chế độ chạy nhanh: phải đặt kèm OUT_DIR / DUCKDB_PATH riêng để không ghi đè kết quả
# thật — dùng `scripts/verify.sh sample`, đừng đặt trong .env.
SAMPLE_PCT = int(os.getenv("SAMPLE_PCT", "100"))
SAMPLE_SEED = os.getenv("SAMPLE_SEED", "0")
if not 1 <= SAMPLE_PCT <= 100:
    raise ValueError("SAMPLE_PCT phải nằm trong [1, 100]")
# "neo4j": đồ thị thật (mặc định). "sql": cùng phép tính bằng DuckDB, không cần Neo4j.
GRAPH_BACKEND = os.getenv("GRAPH_BACKEND", "neo4j").strip().lower()
if GRAPH_BACKEND not in ("neo4j", "sql"):
    raise ValueError("GRAPH_BACKEND phải là 'neo4j' hoặc 'sql'")
# Phép thử có/không: danh sách đặc trưng (cách nhau dấu phẩy, nhận mẫu kiểu peer_*)
# bị bỏ khỏi cả hai model. Rỗng = dùng mọi đặc trưng.
ABLATE_FEATURES = [
    x.strip() for x in os.getenv("ABLATE_FEATURES", "").split(",") if x.strip()
]

GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

DUCKDB_MEMORY = os.getenv("DUCKDB_MEMORY", "3GB")
# DuckDB dùng thêm bộ nhớ ngoài memory_limit theo từng luồng. Với 12 luồng, bước S4
# từng chạm 4,7 GB trên trần 5 GB của container và bị kill ngẫu nhiên (exit 137).
DUCKDB_THREADS = int(os.getenv("DUCKDB_THREADS", "4"))
TMP_DIR = os.path.join(OUT_DIR, "tmp")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def connect(read_only: bool = False):
    """Mở DuckDB với giới hạn bộ nhớ và cho phép tràn ra đĩa.
    Container chỉ có vài GB RAM nên bắt buộc, nếu không sẽ bị OOM (exit 137)."""
    import duckdb

    con = duckdb.connect(DUCKDB_PATH, read_only=read_only)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET threads={DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)
