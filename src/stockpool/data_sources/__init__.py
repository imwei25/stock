"""可插拔数据源后端。

每个后端模块导出以下函数,返回标准化 OHLCV DataFrame
(列: date / open / high / low / close / volume;个股 volume 单位统一为"股"):

    fetch_stock(code: str, start: str | None, min_bars: int | None = None) -> pd.DataFrame
        # min_bars 仅 mootdx 实现(单次 800 根上限,分页凑够);其余后端忽略
    fetch_index(symbol: str) -> pd.DataFrame     # symbol 形如 "sh000001" / "sz399001"

板块(行业)数据无统一后端,始终由 fetcher.py 内的 akshare 路径处理。
"""
