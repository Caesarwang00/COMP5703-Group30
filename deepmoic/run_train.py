# deepmoic/run_train.py
# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parent.parent   # 仓库根: .../COMP5703Group30
PKG  = ROOT / "deepmoic"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    # 默认用 deepmoic/config/gbm_small.yaml；你也可以改成自己的路径
    cfg_path = PKG / "config" / "gbm_small_rna147.yaml"
    print(f"[CONFIG] using: {cfg_path}")

    # 读取成 dict（train 也支持传路径，但读成 dict 更稳）
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from deepmoic.scripts.train import main as train_main
    train_main(cfg)   # 直接把 dict 传进去

if __name__ == "__main__":
    main()
