"""Chạy toàn bộ pipeline: ETL -> trạng thái -> đồ thị -> đặc trưng -> huấn luyện."""

import os
import sys
import time

import s1_etl
import s2_state
import s3_graph
import s3_graph_sql
import s4_features
import s5_train
from config import (
    DATA_DIR,
    DECISION_FREQ,
    GRAPH_BACKEND,
    OUT_DIR,
    SAMPLE_PCT,
    connect,
    log,
)
from validate import REQUIRED, validate_inputs

DONE = os.path.join(OUT_DIR, "_pipeline_done")
STEPS = {
    "s1_etl": s1_etl,
    "s2_state": s2_state,
    "s3_graph": s3_graph,
    "s4_features": s4_features,
    "s5_train": s5_train,
}
# Đổi nhịp quyết định là đổi toàn bộ artifacts, nên nhịp nằm trong mã phiên bản.
PIPELINE_VERSION = f"phs-v6-{DECISION_FREQ}" + (
    f"-mau{SAMPLE_PCT}" if SAMPLE_PCT < 100 else ""
)


def check_data():
    """Báo sớm và rõ nếu thiếu file CSV hoặc sai cột/kiểu dữ liệu, thay vì để
    lỗi khó hiểu ở giữa pipeline."""
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        log(
            "pipeline",
            f"THIẾU {len(missing)}/{len(REQUIRED)} file dữ liệu trong thư mục data/:",
        )
        for f in missing:
            log("pipeline", f"    - {f}")
        log(
            "pipeline",
            f"Đặt đủ {len(REQUIRED)} file CSV vào thư mục data/ rồi chạy lại:",
        )
        log("pipeline", "    scripts/verify.sh full --rerun")
        sys.exit(1)

    errors = validate_inputs(DATA_DIR)
    if errors:
        log("pipeline", f"DỮ LIỆU ĐẦU VÀO có {len(errors)} lỗi cột/kiểu dữ liệu:")
        for e in errors:
            log("pipeline", f"    - {e}")
        log(
            "pipeline", "Sửa các file trên rồi chạy lại: scripts/verify.sh full --rerun"
        )
        sys.exit(1)
    log("pipeline", f"dữ liệu đầu vào hợp lệ ({len(REQUIRED)} file).")


def is_done() -> bool:
    """Artifacts hiện có được dựng bởi đúng phiên bản pipeline này?"""
    try:
        with open(DONE, encoding="utf-8") as f:
            return f.read().strip().split("|")[0] == PIPELINE_VERSION
    except OSError:
        return False


def wait_until_done(poll_seconds: int = 15):
    """App gọi hàm này lúc khởi động: CHỜ kết quả chứ không tự chạy pipeline.

    Chỉ có hai nơi được ghi DuckDB — DAG `pipeline_khuyen_nghi` của Airflow và
    `scripts/verify.sh full --rerun`. Khi app cũng tự chạy, hai tiến trình từng
    tranh nhau khoá file và để lại kết quả dở dang.
    """
    if is_done():
        return
    log("pipeline", "chưa có kết quả của phiên bản hiện tại — app đang CHỜ. Chạy:")
    log("pipeline", "    scripts/verify.sh full --rerun")
    log("pipeline", "hoặc bấm DAG pipeline_khuyen_nghi trong Airflow.")
    while not is_done():
        time.sleep(poll_seconds)
    log("pipeline", "đã có kết quả — tiếp tục khởi động app.")


def run(force: bool = False):
    if is_done() and not force:
        log(
            "pipeline",
            "đã chạy đúng phiên bản hiện tại — bỏ qua (thêm --force để chạy lại).",
        )
        return
    check_data()
    # Mở thử DuckDB để ghi TRƯỚC khi xoá dấu: nếu app đang giữ file thì dừng ngay ở
    # đây, không để lại trạng thái "mất dấu hoàn tất nhưng dữ liệu vẫn nguyên".
    connect().close()
    # Xoá dấu trước khi chạy: nếu lần chạy này hỏng giữa chừng, DuckDB dở dang
    # (bảng mới, model cũ) không được coi là kết quả hợp lệ.
    clear_done()
    t0 = time.time()
    for name in STEPS:
        run_step(name)
    mark_done()
    log("pipeline", f"HOÀN TẤT sau {(time.time() - t0) / 60:.0f} phút")


def run_step(name: str):
    """Chạy đúng một bước — Airflow gọi từng bước để thấy bước nào hỏng, mất bao lâu."""
    s = time.time()
    step = STEPS[name]
    if name == "s3_graph" and GRAPH_BACKEND == "sql":
        step = s3_graph_sql
    step.main()
    log("pipeline", f"{name} xong sau {time.time() - s:.0f}s")


def mark_done():
    with open(DONE, "w", encoding="utf-8") as f:
        f.write(f"{PIPELINE_VERSION}|{time.time()}")


def clear_done():
    """Đánh dấu artifacts đang được dựng lại, để một lần chạy hỏng giữa chừng
    không bị coi là kết quả hợp lệ."""
    if os.path.exists(DONE):
        os.remove(DONE)


def main(argv):
    """python pipeline.py [--force | --check-data | --clear-done | --step TÊN | --mark-done]"""
    if "--check-data" in argv:
        check_data()
    elif "--clear-done" in argv:
        clear_done()
    elif "--mark-done" in argv:
        mark_done()
    elif "--step" in argv:
        name = argv[argv.index("--step") + 1]
        if name not in STEPS:
            sys.exit(f"bước không hợp lệ: {name} (chọn một trong {', '.join(STEPS)})")
        run_step(name)
    else:
        run(force="--force" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
