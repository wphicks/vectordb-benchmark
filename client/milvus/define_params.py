from dataclasses import dataclass, field
from typing import Optional, Any


MILVUS_DEFAULT_COLLECTION = "milvus_benchmark_collection"
MILVUS_DEFAULT_METRIC_TYPE = "L2"
MILVUS_DEFAULT_FIELD_NAME = "float_vector"
MILVUS_DEFAULT_DESCRIPTION = ""
MILVUS_DEFAULT_MAX_LENGTH = 256
MILVUS_DEFAULT_DIM = 128


DEFAULT_PRECISION = 3


@dataclass
class SearchParams:
    vectors: Optional[list] = field(default_factory=lambda: [])


@dataclass
class MilvusPrams:
    database_params: Optional[dict] = field(default_factory=lambda: {})
    connection_params: Optional[dict] = field(default_factory=lambda: {})
    collection_params: Optional[dict] = field(default_factory=lambda: {})
    index_params: Optional[dict] = field(default_factory=lambda: {})
    load_params: Optional[dict] = field(default_factory=lambda: {})
    concurrent_params: Optional[dict] = field(default_factory=lambda: {})
    concurrent_tasks: Optional[list] = field(default_factory=lambda: [])


@dataclass
class MilvusConcurrentParams:
    concurrent_during_time: int
    interval: int
    parallel: int

    # search params
    search_nq: Optional[int] = 0
    search_vectors_len: Optional[int] = 0
    search_params: Optional[dict] = field(default_factory=lambda: {})
    search_other_params: Optional[dict] = field(default_factory=lambda: {})

    # query params
    query_params: Optional[dict] = field(default_factory=lambda: {})
    query_other_params: Optional[dict] = field(default_factory=lambda: {})

