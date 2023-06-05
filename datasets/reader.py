import numpy as np
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
        with np.load('datasets/dataset_files/vdb.npz') as data:
            indexed_vectors = data['indexed_vectors']
        train_len = len(indexed_vectors)
        all_iter = [batch for i in range(batch, train_len, batch)]
        if train_len % batch > 0:
            all_iter += (train_len % batch, )

        _start = 0
        for i in all_iter:
            _end = _start + i
            yield [d for d in range(_start, _end)], indexed_vectors[_start:_end]
            _start = _end

    def iter_test_vectors(self, batch: int, top_k: int):
        v, n, d, b = [], [], [], 0
        with np.load('datasets/dataset_files/vdb.npz') as data:
            search_vectors = data['search_vectors']
            ids = data['ids']
            dists = data['dists']
        for vectors, neighbors, distances in zip(
                search_vectors, ids, dists
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
