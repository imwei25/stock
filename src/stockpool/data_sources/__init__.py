"""可插拔数据源后端。

每个后端模块导出以下函数,返回标准化 OHLCV DataFrame
(列: date / open / high / low / close / volume):

    fetch_stock(code: str, start: str | None) -> pd.DataFrame
    fetch_index(symbol: str) -> pd.DataFrame     # symbol 形如 "sh000001" / "sz399001"

板块(行业)数据无统一后端,始终由 fetcher.py 内的 akshare 路径处理。
"""
