from client.base.client_base import ClientBase


class ClientMock(ClientBase):
    def __init__(self, host: str, params: dict):
        super().__init__()

    def init_db(self, *args, **kwargs):
        pass

    def concurrent_pool_function(self, *args, **kwargs):
        pass
