"""Cấu hình dùng chung — đọc từ biến môi trường (.env)."""

import os

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
OUT_DIR = os.getenv("OUT_DIR", "/app/artifacts")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "/app/artifacts/vienvien.duckdb")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme")  # giá trị thật nằm trong .env

HORIZON = int(os.getenv("HORIZON", "20"))  # số phiên nhìn tới (nhãn)
N_CANDIDATES = int(os.getenv("N_CANDIDATES", "50"))  # ứng viên / khách / tháng
TOPK = int(os.getenv("TOPK", "10"))
COOC_WINDOW = int(os.getenv("COOC_WINDOW", "180"))  # cửa sổ đồng mua (ngày)
COOC_TOPN = int(os.getenv("COOC_TOPN", "25"))  # số mã đồng mua mạnh nhất / mã

# Điểm cuối cho nhánh MUA = MARKET_WEIGHT * điểm nghiên cứu
#                              + (1-MARKET_WEIGHT) * điểm hành vi khách.
MARKET_WEIGHT = float(os.getenv("MARKET_WEIGHT", "0.60"))
BEHAVIOR_HALF_LIFE = int(os.getenv("BEHAVIOR_HALF_LIFE", "90"))
RECENT_WINDOW = int(os.getenv("RECENT_WINDOW", "90"))

if not 0 <= MARKET_WEIGHT <= 1:
    raise ValueError("MARKET_WEIGHT phải nằm trong [0, 1]")

GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

DUCKDB_MEMORY = os.getenv("DUCKDB_MEMORY", "3GB")
TMP_DIR = os.path.join(OUT_DIR, "tmp")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


def connect(read_only: bool = False):
    """Mở DuckDB với giới hạn bộ nhớ và cho phép tràn ra đĩa.
    Container chỉ có vài GB RAM nên bắt buộc, nếu không sẽ bị OOM (exit 137)."""
    import duckdb

    con = duckdb.connect(DUCKDB_PATH, read_only=read_only)
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY}'")
    con.execute(f"SET temp_directory='{TMP_DIR}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)
