import argparse, csv, io, re, sys, time, contextlib, random, copy
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        deep = cur / "deepmoic"
        if deep.is_dir() and (deep / "__init__.py").exists():
            return cur
        cur = cur.parent
    return start


EXEC_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(EXEC_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 训练入口
from deepmoic.scripts.train import main as train_main
import yaml


def autoguess_yaml(name="gbm_small_rna147.yaml"):
    p1 = REPO_ROOT / "deepmoic" / "config" / name
    if p1.exists(): return p1
    for p in (REPO_ROOT / "deepmoic" / "config").glob("*.yaml"):
        return p
    raise FileNotFoundError("找不到基线 YAML（请用 --base 指定）")


def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_by_path(d, dotted, v):
    ks = dotted.split(".")
    cur = d
    for k in ks[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[ks[-1]] = v


def dict_key(d):
    # 用于“组合去重”的规范化 key
    return tuple(sorted((k, str(v)) for k, v in d.items()))


def sample(space, n, seed=2025):
    rnd = random.Random(seed)
    keys = list(space.keys())
    out, seen = [], set()
    while len(out) < n:
        c = {k: rnd.choice(space[k]) for k in keys}
        key = dict_key(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def parse_summary(txt):
    m1 = re.search(r"acc\s*:\s*([0-9\.]+)\s*±\s*([0-9\.]+)", txt)
    m2 = re.search(r"maF1\s*:\s*([0-9\.]+)\s*±\s*([0-9\.]+)", txt)
    out = {}
    if m1:
        out["acc_mean"] = float(m1.group(1))
        out["acc_std"] = float(m1.group(2))
    if m2:
        out["f1_mean"] = float(m2.group(1))
        out["f1_std"] = float(m2.group(2))
    return out


def already_done(row, existing):
    for r in existing:
        same = True
        for k, v in row.items():
            if k in ("trial_id", "ts", "acc_mean", "acc_std", "f1_mean", "f1_std"):
                continue
            if str(r.get(k)) != str(v):
                same = False
                break
        if same:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=None, help="基线 YAML（留空自动找）")
    ap.add_argument("--n", type=int, default=240, help="随机采样组合个数（预算）")
    ap.add_argument("--repeats", type=int, default=20, help="每个 trial 的重复次数")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子（采样）")
    ap.add_argument("--out", type=str, default="sweep_results_mega.csv", help="结果 CSV 路径")
    ap.add_argument("--logdir", type=str, default="sweep_logs_mega", help="日志目录")
    args = ap.parse_args()

    base_yaml = Path(args.base) if args.base else autoguess_yaml()
    base_cfg = load_yaml(base_yaml)

    out_csv = REPO_ROOT / args.out
    logdir = REPO_ROOT / args.logdir
    logdir.mkdir(parents=True, exist_ok=True)

    # ----------------- 搜索空间（大范围，但仍是“合理值域”） -----------------
    SPACE = {
        # --- 特征选择：围绕 8/12 小范围搜索 ---
        "DATA.SELECT_TOP_VAR.per_modality_k.rna": [8, 10, 12],
        "DATA.SELECT_TOP_VAR.per_modality_k.cnv": [10, 12, 15],

        # --- AE（以 512/56/0.30 为中心）---
        "AE.HIDDEN": [384, 512, 640],
        "AE.LATENT_PER_MODAL": [48, 56, 64],
        "AE.DROPOUT": [0.25, 0.30, 0.35],
        "AE.LR": [5e-4, 8e-4, 1e-3],
        "AE.WEIGHT_DECAY": [5e-4, 1e-3],

        # --- GCN（以 192×4、dp=0.45、lr=8e-4 为中心）---
        "GCN.HIDDEN": [160, 192, 224],
        "GCN.LAYERS": [3, 4],
        "GCN.ALPHA": [0.05],  # 已较稳，先固定
        "GCN.DROPOUT": [0.40, 0.45, 0.50],
        "GCN.LR": [8e-4, 1e-3, 1.2e-3],
        "GCN.WEIGHT_DECAY": [3e-4, 5e-4, 7e-4],
        "GCN.LABEL_SMOOTHING": [0.0, 0.01],  # 多数情况下 0.0 更好，但保留 0.01 以防万一
        "GCN.CLASS_WEIGHTED_CE": [True],

        # --- 训练流程 ---
        "GCN.EARLY_STOP": [True],
        "GCN.EARLY_STOP_PATIENCE": [60, 80, 100],
        "EXPERIMENT.TRAIN_RATIO": [0.80, 0.85],  # 0.85 曾经更好，但给 0.80 做对照
        "train.VAL_RATIO": [0.10, 0.15],
    }

    # 固定每个 trial 的 REPEATS/SEEDS（评估更稳）
    FIXED = {
        "EXPERIMENT.REPEATS": args.repeats,
        "EXPERIMENT.RANDOM_SEEDS": list(range(1, args.repeats + 1)),
    }

    # 读取已存在结果（用于跳过重复组合）
    existing = []
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    # 采样
    combos = sample(SPACE, n=args.n, seed=args.seed)

    # 输出 CSV 表头
    header = ["trial_id", "ts"] + list(SPACE.keys()) + ["acc_mean", "acc_std", "f1_mean", "f1_std"]
    write_header = not out_csv.exists()
    with open(out_csv, "a", encoding="utf-8", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=header)
        if write_header:
            w.writeheader()

        for i, comb in enumerate(combos, 1):
            row = {"trial_id": f"T{i:04d}", "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            row.update(comb)
            if already_done(row, existing):
                print(f"[Skip] 已存在组合，跳过 {row['trial_id']}")
                continue

            # 组装配置
            cfg = copy.deepcopy(base_cfg)
            for k, v in FIXED.items():
                set_by_path(cfg, k, v)
            for k, v in comb.items():
                set_by_path(cfg, k, v)

            # 运行一次
            s = io.StringIO()
            try:
                with contextlib.redirect_stdout(s):
                    train_main(cfg)
            except SystemExit:
                pass
            txt = s.getvalue()
            (REPO_ROOT / args.logdir / f"{row['trial_id']}.log").write_text(txt, encoding="utf-8")

            # 解析指标
            metrics = parse_summary(txt)
            row.update(metrics)
            w.writerow(row)
            fout.flush()

            print(f"[Done] {row['trial_id']}  f1={row.get('f1_mean')}±{row.get('f1_std')}  "
                  f"acc={row.get('acc_mean')}±{row.get('acc_std')}")
    print("结果 CSV：", out_csv)
    print("日志目录：", logdir)


if __name__ == "__main__":
    main()