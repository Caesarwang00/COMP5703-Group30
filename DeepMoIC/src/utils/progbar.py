# src/utils/progbar.py
# -*- coding: utf-8 -*-
from contextlib import contextmanager
from typing import Optional, Dict
from tqdm.auto import tqdm

def _should_disable(cfg: Optional[Dict]) -> bool:
    # 和 CFG["USE_TQDM"] 对齐；默认为 True（显示进度）
    return not (cfg or {}).get("USE_TQDM", True)

def _one_line_tqdm(total: int, desc: str, disable: bool, unit: Optional[str] = None):
    bar = tqdm(
        total=total,
        desc=desc,
        position=0,           # 固定同一行
        leave=False,          # 完成后清除该行
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.3,
        bar_format=(
            "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}] {postfix}"
        ),
        disable=disable,
    )
    if unit:
        bar.unit = unit
    return bar

@contextmanager
def one_line_bar(total: int, desc: str, cfg: Optional[Dict] = None, unit: Optional[str] = None):
    """统一的一行进度条（with 语法）"""
    bar = _one_line_tqdm(total=total, desc=desc, disable=_should_disable(cfg), unit=unit)
    try:
        yield bar
    finally:
        bar.close()

def step_postfix(bar, text: str):
    """更新尾部文字，不换行"""
    bar.set_postfix_str(text, refresh=False)

def get_lr(optimizer) -> float:
    for pg in getattr(optimizer, "param_groups", []):
        return float(pg.get("lr", 0.0))
    return 0.0

def patch_tqdm_one_line(enable: bool = True):
    """
    全局把 tqdm 改成单行风格（可选）。
    在程序最开始调用：patch_tqdm_one_line(CFG.get("FSD_PROGRESS_MODE") == "one-line")
    """
    if not enable:
        return
    import tqdm.auto as tauto
    _orig = tauto.tqdm

    def _patched(*args, **kwargs):
        kwargs.setdefault("position", 0)
        kwargs.setdefault("leave", False)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 0.2)
        kwargs.setdefault("smoothing", 0.3)
        return _orig(*args, **kwargs)

    tauto.tqdm = _patched
