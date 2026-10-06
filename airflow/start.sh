#!/usr/bin/env bash
# Khởi động Airflow với tài khoản quản trị lấy từ .env:
#   AIRFLOW_ADMIN_USER      (mặc định: admin)
#   AIRFLOW_ADMIN_PASSWORD  (bắt buộc)
# Đổi mật khẩu trong .env rồi `docker compose up -d airflow` là mật khẩu được cập nhật.
set -euo pipefail

AIRFLOW_ADMIN_USER="${AIRFLOW_ADMIN_USER:-admin}"
if [ -z "${AIRFLOW_ADMIN_PASSWORD:-}" ]; then
    echo "THIẾU AIRFLOW_ADMIN_PASSWORD trong .env — thêm dòng sau rồi chạy lại:" >&2
    echo "    AIRFLOW_ADMIN_PASSWORD=mat_khau_cua_ban" >&2
    sleep 30   # tránh khởi động lại dồn dập trong lúc chờ sửa .env
    exit 1
fi

airflow db migrate

if airflow users list -o plain 2>/dev/null | awk 'NR > 1 {print $2}' | grep -qx "$AIRFLOW_ADMIN_USER"; then
    airflow users reset-password -u "$AIRFLOW_ADMIN_USER" -p "$AIRFLOW_ADMIN_PASSWORD"
else
    airflow users create --role Admin \
        --username "$AIRFLOW_ADMIN_USER" --password "$AIRFLOW_ADMIN_PASSWORD" \
        --firstname Admin --lastname VV --email "${AIRFLOW_ADMIN_EMAIL:-admin@example.com}"
fi

[ "${1:-}" = "--init-only" ] && exit 0

# Chạy cả hai và thoát ngay khi MỘT trong hai dừng, để Docker khởi động lại container.
# Nếu chỉ `exec` webserver thì scheduler chết âm thầm: giao diện vẫn lên nhưng không
# DAG nào chạy nữa.
airflow scheduler &
airflow webserver --port 8080 &
trap 'kill $(jobs -p) 2>/dev/null' TERM INT
wait -n || true
echo "một tiến trình Airflow đã dừng — thoát để container được khởi động lại" >&2
kill $(jobs -p) 2>/dev/null || true
exit 1
