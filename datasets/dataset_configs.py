import os
from typing import Optional
from dataclasses import dataclass
from datasets import DATASET_FILES_DIR


@dataclass
class DatasetConfig:
    name: str
    dim: int
    link: str
    path: Optional[str] = None
    suffix: Optional[str] = None
    similarity_metric_type: Optional[str] = ""


def get_dataset_config(dataset_name: str = "glove-25-angular"):
    config = dataset_configs.get(dataset_name, {})
    config.update({"name": dataset_name,
                   "path": config.get("path", None) or DATASET_FILES_DIR + os.path.split(config.get("link", ""))[-1],
                   "suffix": os.path.splitext(config.get("link", ""))[-1]})
    return DatasetConfig(**config)


dataset_configs = {
    "deep-image-96-angular": {
        "dim": 96,
        "link": "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        "similarity_metric_type": "cosine"
    },
    "gist-960-euclidean": {
        "dim": 960,
        "link": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "similarity_metric_type": "l2"
    },
    "glove-100-angular": {
        "dim": 100,
        "link": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "similarity_metric_type": "cosine"
    },
    "glove-25-angular": {
        "dim": 25,
        "link": "http://ann-benchmarks.com/glove-25-angular.hdf5",
        "similarity_metric_type": "cosine"
    },
}
