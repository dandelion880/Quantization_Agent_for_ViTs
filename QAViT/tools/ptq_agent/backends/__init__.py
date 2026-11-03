"""模型包加载使用的后端工厂；测试夹具由显式 factory 注入。"""
from .base import Backend
from ..schemas import JSON, require


def create_backend(config: JSON) -> Backend:
    require(config["kind"] == "server", "模型包需显式 server 后端；测试夹具使用独立 factory")
    from .server import ServerBackend
    return ServerBackend(config)
