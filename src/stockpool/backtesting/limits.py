"""涨跌停执行约束(C2 / review P1-3)。

回测引擎此前假设 ``open[t]`` 一定可成交;现实中一字涨停开盘买不进单、
一字跌停开盘卖不出。动量/趋势型信号与次日涨停高度正相关,这是买入端的
**正向选择偏差**(收益虚高);跌停卖不出则让回撤被低估。

判定都在比值空间(``open/prev_close``),对 hfq 复权缩放不敏感。
容差 ``_REL_TOL`` 吸收 round-to-cent:开盘价达到理论涨停价 99.8% 即视为
封板(保守 —— 宁可错拒,不可乐观成交)。

完整的盘中触板/部分成交建模(mask_exec)是长期项,这里只消掉最大头的
一字板偏差。
"""
from __future__ import annotations

_REL_TOL = 0.002  # 相对容差 0.2%


def infer_limit_pct(code: str | None, st_codes: "set[str] | None" = None) -> float:
    """按代码前缀推断涨跌停幅度。板块优先于 ST(创业板/科创板 ST 仍 20%)。"""
    if code:
        if code.startswith(("300", "301", "688")):
            return 0.20
        if code.startswith(("82", "83", "87", "43")):
            return 0.30
        if st_codes and code in st_codes:
            return 0.05
    return 0.10


def open_hits_limit_up(open_t: float, prev_close: float, limit_pct: float) -> bool:
    """开盘价触及涨停价(买不进)。"""
    if prev_close <= 0:
        return False
    return open_t >= prev_close * (1.0 + limit_pct) * (1.0 - _REL_TOL)


def open_hits_limit_down(open_t: float, prev_close: float, limit_pct: float) -> bool:
    """开盘价触及跌停价(卖不出)。"""
    if prev_close <= 0:
        return False
    return open_t <= prev_close * (1.0 - limit_pct) * (1.0 + _REL_TOL)
