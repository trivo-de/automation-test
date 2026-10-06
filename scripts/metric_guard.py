"""Chốt chặn chỉ số: lưu kết quả mỗi lần chạy và báo khi chất lượng model tụt.

Mỗi lần gọi sẽ ghi một dòng vào artifacts/metrics_history.jsonl rồi so với lần
chạy trước đó (cùng nhịp quyết định). Thoát mã 1 nếu HitRate@1 của model hành vi
giảm quá METRIC_GUARD_MAX_DROP, hoặc model không còn hơn mức chọn ngẫu nhiên.

Chỉ cần thư viện chuẩn, chạy được ở bất kỳ đâu có thư mục artifacts/:
    python scripts/metric_guard.py [thư_mục_artifacts]
"""

import datetime as dt
import json
import os
import sys

MAX_DROP = float(os.getenv("METRIC_GUARD_MAX_DROP", "0.02"))
TRACKED = ("buy", "portfolio")


def snapshot(summary):
    row = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "decision_freq": summary.get("decision_freq"),
        "market_weight": summary.get("market_weight"),
    }
    for name in TRACKED:
        r = summary.get(name) or {}
        row[name] = {
            "auc": r.get("auc"),
            "random_hit@1": r.get("random_hit@1"),
            "behavior_hit@1": (r.get("behavior") or {}).get("hit@1"),
            "final_hit@1": (r.get("final") or {}).get("hit@1"),
            "n_groups": r.get("n_groups"),
        }
    return row


def compare(current, previous, max_drop=MAX_DROP):
    """Trả về danh sách vi phạm (rỗng nếu ổn). `previous` có thể là None."""
    problems = []
    for name in TRACKED:
        now = current[name]
        hit, rnd = now["behavior_hit@1"], now["random_hit@1"]
        if hit is None or rnd is None:
            problems.append(f"{name}: thiếu kết quả đánh giá")
            continue
        if hit <= rnd:
            problems.append(f"{name}: HitRate@1 {hit} không hơn chọn ngẫu nhiên {rnd}")
        before = (previous or {}).get(name, {}).get("behavior_hit@1")
        if before is not None and hit < before - max_drop:
            problems.append(
                f"{name}: HitRate@1 giảm từ {before} xuống {hit} "
                f"(quá ngưỡng {max_drop})"
            )
    return problems


def main(out_dir):
    with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as f:
        current = snapshot(json.load(f))

    history_path = os.path.join(out_dir, "metrics_history.jsonl")
    history = []
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            history = [json.loads(line) for line in f if line.strip()]
    # chỉ so với lần chạy cùng nhịp quyết định, nếu không hai con số không so được
    # ... và chỉ so với lần chạy ĐÃ QUA: nếu lấy cả lần đỏ làm mốc thì chạy lại mà
    # không sửa gì sẽ thành xanh, và tụt dần từng chút sẽ không bao giờ bị bắt.
    comparable = [
        h
        for h in history
        if h.get("decision_freq") == current["decision_freq"] and h.get("passed")
    ]
    previous = comparable[-1] if comparable else None

    problems = compare(current, previous)
    current["passed"] = not problems
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(current, ensure_ascii=False) + "\n")

    print(f"{'nhánh':10} {'chỉ số':16} {'lần trước':>10} {'lần này':>9}")
    for name in TRACKED:
        for key in ("auc", "random_hit@1", "behavior_hit@1", "final_hit@1", "n_groups"):
            before = (previous or {}).get(name, {}).get(key)
            print(
                f"{name:10} {key:16} {before if before is not None else '—':>10} "
                f"{current[name][key] if current[name][key] is not None else '—':>9}"
            )
    print(f"đã ghi lần chạy thứ {len(history) + 1} vào {history_path}")
    if problems:
        print("CHỈ SỐ TỤT: " + "; ".join(problems))
        sys.exit(1)
    print("chỉ số ổn.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.getenv("OUT_DIR", "artifacts"))
