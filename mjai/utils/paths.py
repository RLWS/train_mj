from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回路径对象。
    
    参数:
        path (str | Path): 文件或目录路径。
    
    返回:
        Path: 函数执行结果。
    
    异常:
        无。
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
