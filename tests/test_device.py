from __future__ import annotations

from mjai.utils.device import normalize_device_name


def test_normalize_p800_device_alias():
    """验证P800 设备别名规范化。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    assert normalize_device_name("p800") == "xpu"
    assert normalize_device_name("xpu") == "xpu"
