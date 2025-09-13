import re

def to_short_sample_id(s: str) -> str:
    """
    统一到 'TCGA-XX-XXXX-YY'（含样本类型两位，如 01/11）。
    '.'→'-'；大写；保留前 4 段；01A/01B 会归并到 01。
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.replace('.', '-').upper()
    parts = s.split('-')
    if len(parts) >= 4:
        return '-'.join(parts[:4])
    return s

def is_primary_tumor(s: str) -> bool:
    s = to_short_sample_id(s)
    parts = s.split('-')
    if len(parts) < 4:
        return False
    return parts[3] == '01'
