from __future__ import annotations


def normalize_device_name(device: str | None) -> str:
    """规范化用户输入的设备名称。
    
    参数:
        device (str | None): 计算设备。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    if device is None or device == "auto":
        return "auto"
    normalized = device.lower()
    aliases = {
        "gpu": "cuda",
        "cuda:0": "cuda",
        "p800": "xpu",
    }
    return aliases.get(normalized, normalized)
