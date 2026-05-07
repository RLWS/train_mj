from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """加载 YAML 文件并返回配置字典。
    
    参数:
        path (str | Path): 文件或目录路径。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        TypeError: 当输入类型不符合要求时抛出。
    """
    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def load_resolved_yaml(path: str | Path) -> dict[str, Any]:
    """加载配置文件并解析默认配置与占位符。
    
    参数:
        path (str | Path): 文件或目录路径。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        TypeError: 当输入类型不符合要求时抛出。
    """
    config_path = Path(path)
    root = load_yaml(config_path)
    defaults = root.pop("defaults", [])

    resolved: dict[str, Any] = {}
    for default in defaults:
        if not isinstance(default, dict) or len(default) != 1:
            raise TypeError(f"defaults entries must be one-key mappings: {config_path}")
        section_name, config_name = next(iter(default.items()))
        section_path = config_path.parent / section_name / f"{config_name}.yaml"
        resolved[section_name] = load_yaml(section_path)

    resolved = _merge_dicts(resolved, root)
    return _interpolate_top_level_strings(resolved)


def dump_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """将配置字典写入 YAML 文件。
    
    参数:
        path (str | Path): 文件或目录路径。
        data (dict[str, Any]): 函数执行所需的输入参数。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        base (dict[str, Any]): 函数执行所需的输入参数。
        update (dict[str, Any]): 函数执行所需的输入参数。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        无。
    """
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _interpolate_top_level_strings(config: dict[str, Any]) -> dict[str, Any]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        config (dict[str, Any]): 配置字典。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        无。
    """
    interpolated: dict[str, Any] = {}
    replacements = {
        key: str(value)
        for key, value in config.items()
        if isinstance(value, (str, int, float))
    }

    for key, value in config.items():
        if isinstance(value, str):
            interpolated[key] = _replace_placeholders(value, replacements)
        elif isinstance(value, dict):
            interpolated[key] = _interpolate_nested_dict(value, replacements)
        else:
            interpolated[key] = value
    return interpolated


def _interpolate_nested_dict(data: dict[str, Any], replacements: dict[str, str]) -> dict[str, Any]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        data (dict[str, Any]): 函数执行所需的输入参数。
        replacements (dict[str, str]): 相关数据列表。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        无。
    """
    resolved: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            resolved[key] = _replace_placeholders(value, replacements)
        elif isinstance(value, dict):
            resolved[key] = _interpolate_nested_dict(value, replacements)
        else:
            resolved[key] = value
    return resolved


def _replace_placeholders(value: str, replacements: dict[str, str]) -> str:
    """执行与函数名对应的业务逻辑。
    
    参数:
        value (str): 输入值。
        replacements (dict[str, str]): 相关数据列表。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    resolved = value
    for key, replacement in replacements.items():
        resolved = resolved.replace(f"${{{key}}}", replacement)
    return resolved
