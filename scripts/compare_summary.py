"""So kết quả chạy nhanh (mẫu khách) với kết quả chạy đầy đủ.

    python scripts/compare_summary.py artifacts/quick/summary.json artifacts/summary.json

Chỉ dùng thư viện chuẩn. In HitRate@1 của hai bên kèm sai số chuẩn của bản chạy nhanh
(tính theo cụm khách, do s5_train ghi vào summary.json), để biết một chênh lệch là thật
hay chỉ là nhiễu.
"""

import json
import sys

ROWS = (
    ("chọn ngẫu nhiên", None),
    ("chỉ tín hiệu PHS", "baseline"),
    ("chỉ model hành vi", "behavior"),
    ("điểm cuối", "final"),
)


def hit(result, variant):
    if variant is None:
        return result.get("random_hit@1")
    return (result.get(variant) or {}).get("hit@1")


def standard_error(result, variant):
    """Sai số chuẩn theo cụm khách. Không tự tính từ số nhóm: các nhóm của cùng một
    khách ở các tuần liền nhau không độc lập, công thức nhị thức sẽ cho số quá nhỏ."""
    if variant is None:
        return None
    return (result.get(variant) or {}).get("hit@1_se")


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}"


def main(quick_path, full_path=None):
    with open(quick_path, encoding="utf-8") as f:
        quick = json.load(f)
    full = {}
    if full_path:
        try:
            with open(full_path, encoding="utf-8") as f:
                full = json.load(f)
        except OSError:
            pass

    label = f"mẫu {quick.get('sample_pct', '?')}% khách, đồ thị {quick.get('graph_backend')}"
    if quick.get("ablate_features"):
        label += f", bỏ {','.join(quick['ablate_features'])}"
    print(f"Chạy nhanh: {label}")
    print(
        f"{'nhánh':10} {'cách chấm điểm':20} {'nhanh':>7} {'± sai số':>9} {'đầy đủ':>8}"
    )
    for name, title in (("buy", "MUA"), ("portfolio", "DANH MỤC")):
        q = quick.get(name) or {}
        r = full.get(name) or {}
        n = q.get("n_groups")
        for row, variant in ROWS:
            p = hit(q, variant)
            print(
                f"{title:10} {row:20} {fmt(p):>7} {fmt(standard_error(q, variant)):>9} "
                f"{fmt(hit(r, variant)):>8}"
            )
        print(
            f"{title:10} {'số nhóm xếp hạng được':20} {n if n is not None else '—':>7}"
            f" {'':>9} {r.get('n_groups', '—'):>8}"
        )
    print(
        "Sai số tính theo cụm khách. Chênh lệch nhỏ hơn 2 lần sai số thì coi là nhiễu.\n"
        "Chỉ so hai lần chạy trên CÙNG một mẫu; mức điểm của mẫu không so được với bản đầy đủ."
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
