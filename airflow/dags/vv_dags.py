"""DAG Airflow của dự án — thay cho CI trên GitHub.

  ci_kiem_tra_code      cứ 5 phút xem có commit mới không; có thì chạy lint, kiểm tra
                        định dạng và bộ test (dữ liệu giả, không cần Neo4j).
  pipeline_khuyen_nghi  bấm chạy tay: kiểm tra dữ liệu -> S1..S5 từng bước -> bật lại
                        app -> kiểm tra hệ thống thật -> chốt chặn chỉ số.

Airflow không chứa thư viện của dự án. Mỗi task chạy trong một container riêng tạo
từ image của app (DockerOperator), nên môi trường giống hệt lúc chạy tay bằng
scripts/verify.sh.
"""

import os
import re

import pendulum
from airflow import DAG
from airflow.models import Variable
from airflow.models.baseoperator import chain
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Đường dẫn dự án TRÊN MÁY CHỦ: DockerOperator nhờ Docker của máy chủ gắn thư mục,
# nên không dùng được đường dẫn bên trong container Airflow.
HOST_DIR = os.environ["HOST_PROJECT_DIR"]
PROJECT_IN_AIRFLOW = "/opt/project"  # cùng thư mục đó, gắn chỉ-đọc vào Airflow
IMAGE = os.getenv("PIPELINE_IMAGE", "personalized_stock_recommendation-app")
NETWORK = os.getenv("PIPELINE_NETWORK", "personalized_stock_recommendation_default")
DOCKER_URL = os.getenv("DOCKER_URL", "tcp://docker-proxy:2375")
APP_CONTAINER = os.getenv("APP_CONTAINER", "vv-app")
START = pendulum.datetime(2026, 10, 1, tz="Asia/Ho_Chi_Minh")


def project_env():
    """Chuyển tiếp cho task đúng các biến mà .env.example khai báo.

    Giá trị lấy từ môi trường của container Airflow (docker-compose nạp .env vào
    đó), nên thêm một tham số mới vào .env.example là task tự nhận được.
    """
    names = []
    try:
        with open(f"{PROJECT_IN_AIRFLOW}/.env.example", encoding="utf-8") as f:
            names = re.findall(r"^([A-Z][A-Z0-9_]*)=", f.read(), flags=re.M)
    except OSError:
        pass
    # Tài khoản Airflow chỉ dành cho Airflow, không đưa vào container của task.
    return {
        k: os.environ[k]
        for k in names
        if k in os.environ and not k.startswith("AIRFLOW_")
    }


def in_app_image(task_id, command, mounts, **kwargs):
    return DockerOperator(
        task_id=task_id,
        image=IMAGE,
        command=["sh", "-c", command],
        docker_url=DOCKER_URL,
        network_mode=NETWORK,
        mounts=mounts,
        environment=project_env() | kwargs.pop("environment", {}),
        mount_tmp_dir=False,
        auto_remove="force",
        mem_limit="5g",
        **kwargs,
    )


# ------------------------------------------------------------------ CI
def has_new_commit(dag_run=None):
    """True nếu HEAD đã đổi từ lần kiểm tra trước, hoặc khi được bấm chạy tay."""
    git = f"{PROJECT_IN_AIRFLOW}/.git"
    with open(f"{git}/HEAD", encoding="utf-8") as f:
        head = f.read().strip()
    sha = head
    if head.startswith("ref: "):
        ref = head[5:]
        try:
            with open(f"{git}/{ref}", encoding="utf-8") as f:
                sha = f.read().strip()
        except OSError:  # ref đã được gom vào packed-refs
            with open(f"{git}/packed-refs", encoding="utf-8") as f:
                sha = next((x.split()[0] for x in f if x.strip().endswith(ref)), head)

    last = Variable.get("ci_last_sha", default_var="")
    manual = dag_run is not None and dag_run.run_type == "manual"
    print(f"HEAD = {sha} | lần kiểm tra trước = {last or '(chưa có)'}")
    if sha == last and not manual:
        return False
    # Ghi ngay để mỗi commit chỉ được kiểm tra một lần, kể cả khi kiểm tra đỏ.
    Variable.set("ci_last_sha", sha)
    return True


WORK = [Mount(source=HOST_DIR, target="/work", type="bind")]
PATHS = "src tests scripts airflow"

with DAG(
    dag_id="ci_kiem_tra_code",
    description="Lint + định dạng + test mỗi khi có commit mới",
    schedule="*/5 * * * *",
    start_date=START,
    catchup=False,
    max_active_runs=1,
    tags=["ci"],
) as ci:
    detect = ShortCircuitOperator(
        task_id="co_commit_moi", python_callable=has_new_commit
    )
    lint = in_app_image("lint", f"ruff check {PATHS}", WORK, working_dir="/work")
    fmt = in_app_image(
        "dinh_dang", f"ruff format --check {PATHS}", WORK, working_dir="/work"
    )
    test = in_app_image(
        "test",
        "pip install -q --disable-pip-version-check --root-user-action=ignore "
        "pytest==8.3.3 && python -m pytest -q -p no:cacheprovider",
        WORK,
        working_dir="/work",
    )
    detect >> lint >> fmt >> test


# ------------------------------------------------------------ PIPELINE
def set_app_running(running: bool):
    """Dừng / bật container app qua Docker API.

    Phải dừng app trước khi chạy pipeline: app đang giữ file DuckDB, tiến trình
    khác không mở để ghi được.
    """
    import docker

    container = docker.DockerClient(base_url=DOCKER_URL).containers.get(APP_CONTAINER)
    if running:
        container.start()
    else:
        container.stop(timeout=30)
    print(f"{APP_CONTAINER}: {'đã bật' if running else 'đã dừng'}")


APP_MOUNTS = [
    Mount(source=f"{HOST_DIR}/data", target="/app/data", type="bind", read_only=True),
    Mount(source=f"{HOST_DIR}/src", target="/app/src", type="bind"),
    Mount(source=f"{HOST_DIR}/artifacts", target="/app/artifacts", type="bind"),
    Mount(
        source=f"{HOST_DIR}/scripts", target="/app/scripts", type="bind", read_only=True
    ),
]


def pipeline_cmd(args):
    return f"python -u /app/src/pipeline.py {args}"


with DAG(
    dag_id="pipeline_khuyen_nghi",
    description="Chạy lại toàn bộ pipeline khuyến nghị rồi kiểm tra kết quả",
    schedule=None,
    start_date=START,
    catchup=False,
    max_active_runs=1,
    tags=["pipeline"],
) as pipeline:
    # Kiểm tra dữ liệu TRƯỚC khi dừng app: dữ liệu hỏng thì app vẫn phục vụ bản cũ.
    check_data = in_app_image(
        "kiem_tra_du_lieu", pipeline_cmd("--check-data"), APP_MOUNTS
    )
    stop_app = PythonOperator(
        task_id="dung_app",
        python_callable=set_app_running,
        op_kwargs={"running": False},
    )
    clear = in_app_image("xoa_dau_hoan_tat", pipeline_cmd("--clear-done"), APP_MOUNTS)
    steps = [
        in_app_image(name, pipeline_cmd(f"--step {name}"), APP_MOUNTS)
        for name in ("s1_etl", "s2_state", "s3_graph", "s4_features", "s5_train")
    ]
    mark = in_app_image("ghi_dau_hoan_tat", pipeline_cmd("--mark-done"), APP_MOUNTS)
    # Chỉ bật lại app khi mọi bước đều xong; pipeline hỏng thì app nằm yên để
    # không phục vụ kết quả dở dang.
    start_app = PythonOperator(
        task_id="bat_app", python_callable=set_app_running, op_kwargs={"running": True}
    )
    verify = in_app_image(
        "kiem_tra_he_thong",
        "python -u /app/scripts/verify_app.py",
        APP_MOUNTS,
        environment={
            "VERIFY_APP_URL": f"http://{APP_CONTAINER}:7860/",
            "VERIFY_APP_WAIT": "300",
        },
    )
    guard = in_app_image(
        "chot_chan_chi_so",
        "python -u /app/scripts/metric_guard.py /app/artifacts",
        APP_MOUNTS,
    )

    chain(check_data, stop_app, clear, *steps, mark, start_app, verify, guard)
