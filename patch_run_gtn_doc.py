from pathlib import Path

p = Path("experiments/run_gtn_baseline.py")
s = p.read_text(encoding="utf-8")
old = """    D-gtn-n-epochs  实测 247 名被试（1 名 Experiment_611 损坏），每被试刺激事件 mean=206（范围
                    58–372），**无一人达 review v4 声称的 500**。每数字平均试次 K≈23（非 50）。
                    这直接改写命中率锚点的预期：77%±3 是在 K≈23 下取得的，验收口径须据此校准。"""
new = """    D-gtn-n-epochs  实测 247 名被试可读（Experiment_611 是缺 .txt thought 元数据，非 HDF5 损坏）；
                    原始数字事件 mean≈205（范围 58–372），**无一人达 review v4 声称的 500**。
                    默认 ±150μV 伪迹剔除后另有 3 名被试 0 试次（质量排除），可评估 244 名，
                    每数字平均试次 K≈17（非 50）。77%±3 锚点须按该 K 校准（review v6 实测）。"""
if old not in s:
    # Try variant with existing exact text from file around marker
    i = s.find("D-gtn-n-epochs")
    j = s.find("D-gtn-base-rate")
    print("OLD BLOCK REPR:", repr(s[i:j]))
    raise SystemExit("old not found")
p.write_text(s.replace(old, new), encoding="utf-8")
print("patched")
