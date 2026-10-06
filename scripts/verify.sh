#!/usr/bin/env bash
# Kiểm tra dự án bằng MỘT lệnh, theo trình tự Launch -> Doctor -> Drive -> Evidence.
#
#   scripts/verify.sh quick          lint + test trên dữ liệu giả (vài giây, không cần Neo4j)
#   scripts/verify.sh full           quick + kiểm tra hệ thống thật đang chạy
#   scripts/verify.sh full --rerun   ép chạy lại toàn bộ pipeline rồi mới kiểm tra (~10 phút)
#
#   scripts/verify.sh sample [%]     pipeline thật trên mẫu khách (mặc định 20%), không cần
#                                    Neo4j, không đụng kết quả chính; in HitRate@1 so với
#                                    lần chạy đầy đủ. Thử một ý tưởng: ABLATE_FEATURES='peer_*'
#
# "full" không chỉ xem app có lên hay không: nó gọi hàm gợi ý cho một khách thật
# rồi ĐỌC LẠI dữ liệu gốc để đối chiếu. Bằng chứng ghi vào artifacts/verify_report.json.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-quick}"
RERUN="${2:-}"
PORT="${GRADIO_PORT:-7860}"
WAIT_SECONDS="${VERIFY_WAIT_SECONDS:-300}"

step() { printf '\n== %s ==\n' "$1"; }
die()  { printf 'LỖI: %s\n' "$1" >&2; exit 1; }

quick() {
    step "Lint + test (dữ liệu giả, không cần Neo4j)"
    docker compose run --rm --no-deps -T -v "$PWD:/work" -w /work app sh -c '
        pip install -q --disable-pip-version-check --root-user-action=ignore pytest==8.3.3 &&
        ruff check src tests scripts airflow &&
        ruff format --check src tests scripts airflow &&
        python -m pytest -q'
}

wait_for_app() {
    local start restarts now
    start=$(date +%s)
    restarts=$(docker inspect -f '{{.RestartCount}}' vv-app)
    until curl -sf -o /dev/null "http://localhost:${PORT}/"; do
        # App đứng chờ khi kết quả thuộc phiên bản khác (đổi DECISION_FREQ, tăng phiên
        # bản pipeline): báo ngay cách xử lý thay vì đợi hết giờ.
        if docker logs --tail 3 vv-app 2>&1 | grep -q 'bấm DAG pipeline_khuyen_nghi'; then
            die "app đang chờ kết quả của phiên bản hiện tại — chạy: scripts/verify.sh full --rerun"
        fi
        [ "$(docker inspect -f '{{.State.Running}}' vv-app)" = "true" ] ||
            die "container vv-app đã dừng — xem: docker compose logs --tail 80 app"
        [ "$(docker inspect -f '{{.RestartCount}}' vv-app)" = "$restarts" ] ||
            die "vv-app khởi động lại liên tục — xem: docker compose logs --tail 80 app"
        now=$(date +%s)
        [ $((now - start)) -lt "$WAIT_SECONDS" ] ||
            die "quá ${WAIT_SECONDS}s mà app chưa phục vụ ở cổng ${PORT}"
        sleep 10
    done
}

full() {
    [ -f .env ] || die "thiếu .env — tạo từ .env.example trước"

    step "Launch: Neo4j + app + Airflow"
    docker compose up -d
    if [ "$RERUN" = "--rerun" ]; then
        # App chỉ phục vụ, không tự chạy pipeline. Dừng app (nó giữ file DuckDB),
        # chạy pipeline trong một container riêng, rồi bật app lại.
        step "Chạy lại toàn bộ pipeline (~10 phút)"
        docker compose stop app
        docker compose run --rm --no-deps -T app python -u /app/src/pipeline.py --force |
            grep -E '\[pipeline\]|CẢNH BÁO|nhóm xếp hạng|audit' ||
            die "pipeline hỏng — app được để ở trạng thái dừng; sửa lỗi rồi chạy lại"
        docker compose start app
    fi
    [ -f artifacts/_pipeline_done ] ||
        die "chưa có kết quả pipeline — chạy: scripts/verify.sh full --rerun"
    echo "chờ app phục vụ ở cổng ${PORT}..."
    wait_for_app

    step "Doctor + Drive + Evidence: kiểm tra hệ thống thật"
    docker compose exec -T app python - < scripts/verify_app.py

    step "Doctor: đồ thị Neo4j khớp bản SQL"
    docker compose exec -T app python - < scripts/compare_graph.py

    step "Doctor: DAG Airflow nạp không lỗi"
    tries=0
    until curl -sf -o /dev/null "http://localhost:8080/health"; do
        tries=$((tries + 1))
        [ "$tries" -lt 36 ] ||
            die "Airflow chưa lên sau 3 phút — xem: docker compose logs --tail 20 airflow (thiếu AIRFLOW_ADMIN_PASSWORD trong .env?)"
        sleep 5
    done
    # Lệnh này thoát mã 1 khi CÓ lỗi nạp; `|| true` để còn kịp in nội dung lỗi ra.
    errors=$(docker compose exec -T airflow airflow dags list-import-errors 2>/dev/null) || true
    echo "$errors" | grep -q "No data found" || die "DAG lỗi khi nạp: $errors"
    echo "  [OK ] không có DAG nào lỗi khi nạp"
}

sample() {
    local pct="${1:-20}" start
    start=$(date +%s)
    step "Pipeline trên mẫu ${pct}% khách (đồ thị tính bằng SQL, kết quả ở artifacts/quick)"
    # Xoá kết quả lần trước và giữ mã thoát của pipeline: nếu lần này hỏng thì phải
    # báo hỏng, không được in lại số của lần chạy cũ như thể là số mới.
    mkdir -p artifacts/quick
    rm -f artifacts/quick/summary.json
    local log=artifacts/quick/run.log status=0
    docker compose run --rm --no-deps -T \
        -e SAMPLE_PCT="$pct" -e SAMPLE_SEED="${SAMPLE_SEED:-0}" -e GRAPH_BACKEND=sql \
        -e ABLATE_FEATURES="${ABLATE_FEATURES:-}" \
        -e OUT_DIR=/app/artifacts/quick -e DUCKDB_PATH=/app/artifacts/quick/quick.duckdb \
        app python -u /app/src/pipeline.py --force > "$log" 2>&1 || status=$?
    grep -E 'CHẾ ĐỘ|PHÉP THỬ|pipeline\]|nhóm xếp hạng' "$log" || true
    if [ "$status" -ne 0 ] || [ ! -f artifacts/quick/summary.json ]; then
        tail -15 "$log" >&2
        die "pipeline trên mẫu hỏng (mã thoát $status) — log đầy đủ: $log"
    fi
    echo
    python3 scripts/compare_summary.py artifacts/quick/summary.json artifacts/summary.json
    echo "thời gian: $(( $(date +%s) - start )) giây"
}

command -v docker >/dev/null || die "không tìm thấy docker"
docker info >/dev/null 2>&1 || die "Docker chưa chạy"

case "$MODE" in
    quick) quick ;;
    full)  quick; full ;;
    sample) sample "${2:-20}"; exit 0 ;;
    *)     die "chế độ không hợp lệ: $MODE (dùng quick | full [--rerun] | sample [%])" ;;
esac

step "XONG — mọi kiểm tra đều qua"
