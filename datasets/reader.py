from dataclasses import dataclass
from common.common_func import read_ann_hdf5_file
from datasets.dataset_configs import get_dataset_config
from datasets.dataset_download import DatasetDownload


read_suffix_file = {
    ".hdf5": read_ann_hdf5_file
}


@dataclass
class DatasetContent:
    neighbors: any
    test: any
    train: any
    distances: any


@dataclass
class Search:
    vectors: any
    neighbors: any
    distances: any


class ReaderBase:
    def __init__(self, dataset_name: str = "glove-25-angular"):
        self.config = get_dataset_config(dataset_name)
        self.dataset_content = None

    def get_dataset_content(self):
        DatasetDownload(self.config).download()

        f = read_suffix_file.get(self.config.suffix)(self.config.path)
        self.dataset_content = DatasetContent(f["neighbors"], f["test"], f["train"], f["distances"])

    def iter_train_vectors(self, batch: int):
        all_iter = [batch for i in range(batch, len(self.dataset_content.train), batch)]
        if len(self.dataset_content.train) % batch > 0:
            all_iter += (len(self.dataset_content.train) % batch, )

        _start = 0
        for i in all_iter:
            _end = _start + i
            yield [d for d in range(_start, _end)], self.dataset_content.train[_start:_end]
            _start = _end

    def iter_test_vectors(self, batch: int, top_k: int):
        v, n, d, b = [], [], [], 0
        for vectors, neighbors, distances in zip(
                self.dataset_content.test, self.dataset_content.neighbors, self.dataset_content.distances
        ):
            v.append(vectors.tolist())
            n.append(neighbors.tolist()[:top_k])
            d.append(distances.tolist()[:top_k])
            b += 1
            if b == batch:
                yield Search(v, n, d)
                v, n, d, b = [], [], [], 0
        if b > 0:
            yield Search(v, n, d)
