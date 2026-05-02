from __future__ import annotations

from mjai.utils.device import normalize_device_name


def test_normalize_p800_device_alias():
    assert normalize_device_name("p800") == "xpu"
    assert normalize_device_name("xpu") == "xpu"
