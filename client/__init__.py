from client.milvus.client_milvus import ClientMilvus
from client.mock.client_mock import ClientMock

ClientObject = {
    "milvus": ClientMilvus,
    "mock": ClientMock
}


class ClientEntry:
    def __init__(self, engine: str, host: str, params: dict):
        self.host = host
        self.params = params

        self.client = ClientObject[engine](host=self.host, params=self.params)

    def init_db_client(self):
        self.client.init_db()

    def get_concurrent_start_params(self):
        return self.client.get_concurrent_start_params()
