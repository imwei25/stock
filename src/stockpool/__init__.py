__version__ = "0.1.0"

# pyecharts 默认 CDN (assets.pyecharts.org) 在部分网络下不可达，
# 渲染出的 HTML 加载不到 echarts.min.js 导致图表空白。改用国内可达的 BootCDN。
from pyecharts.globals import CurrentConfig as _CurrentConfig
_CurrentConfig.ONLINE_HOST = "https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/"
