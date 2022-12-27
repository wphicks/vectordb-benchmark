from client.milvus.client_milvus import ClientMilvus
from client.mock.client_mock import ClientMock
from concurrency.multi_process import MultiProcessConcurrent
from datasets.reader import ReaderBase


ClientObject = {
    "milvus": ClientMilvus,
    "mock": ClientMock
}


class ClientEntry:
    def __init__(self, engine: str, host: str, params: dict):
        self.host = host
        self.params = params

        self.client = ClientObject[engine](host=self.host, params=self.params)

    def start_concurrency(self):
        self.client.get_concurrent_start_params()
        MultiProcessConcurrent().start(self.client)

    def start_recall(self, dataset_name, prepare=True):
        rb = ReaderBase(dataset_name=dataset_name)
        rb.get_dataset_content()

        self.client.get_serial_start_params(rb)
        self.client.serial_prepare_data(prepare=prepare)
        self.client.serial_search_recall()
