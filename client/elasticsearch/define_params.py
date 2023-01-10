from dataclasses import dataclass, field
from typing import Optional


ES_DEFAULT_HOST = "localhost"
ES_DEFAULT_PORT = 9200
ES_USER = "elastic"
ES_PASSWORD = "passwd"
ES_DEFAULT_INDEX = "elasticsearch_benchmark_index"
ES_DEFAULT_INDEX_NAME = "hnsw"
ES_DEFAULT_INDEX_PARAMS = {"m": 8, "ef_construction": 200}
ES_DEFAULT_METRIC_TYPE = "l2_norm"  # support l2_norm,dot_product,cosine
ES_DEFAULT_FIELD_NAME = "float_vector"
ES_DEFAULT_DESCRIPTION = ""
ES_DEFAULT_TEXT_LENGTH = 256
ES_DEFAULT_DIM = 128
ES_DEFAULT_NUM_CANDIDATES = 100

DEFAULT_PRECISION = 3


class IndexSupport:
    Long = "long"
    Integer = "integer"
    Short = "short"
    Byte = "byte"
    Float = "float"
    Double = "double"
    Text = "text"
    DenseVector = "dense_vector"


ES_DEFAULT_TYPE = "float"
INDEX_TYPE_MAPPING = {
    # A signed 64-bit integer with a minimum value of -263 and a maximum value of 263-1
    "int64": IndexSupport.Long,
    # A signed 32-bit integer with a minimum value of -231 and a maximum value of 231-1.
    "int32": IndexSupport.Integer,
    # A signed 16-bit integer with a minimum value of -32,768 and a maximum value of 32,767.
    "int16": IndexSupport.Short,
    # A signed 8-bit integer with a minimum value of -128 and a maximum value of 127.
    "int8": IndexSupport.Byte,
    # A single-precision 32-bit IEEE 754 floating point number, restricted to finite values.
    "float": IndexSupport.Float,
    # A double-precision 64-bit IEEE 754 floating point number, restricted to finite values.
    "double": IndexSupport.Double,
    # the traditional field type for full-text content such as the body of an email or the description of a product.
    "text": IndexSupport.Text,
    # A k-nearest neighbor (kNN) search finds the k nearest vectors to a query vector,
    # as measured by a similarity metric.
    "float_vector": IndexSupport.DenseVector,
}

INDEX_OPTIONS_MAPPING = {
    IndexSupport.Long: {"index": True},
    IndexSupport.Integer: {"index": True},
    IndexSupport.Short: {"index": True},
    IndexSupport.Byte: {"index": True},
    IndexSupport.Float: {"index": True},
    IndexSupport.Double: {"index": True},
    IndexSupport.Text: {"index": True},
}


class HttpCode:
    Continue = 100
    OK = 200
    BadRequest = 400
    Forbidden = 403
    NotFound = 404
    BadGateway = 502
    GatewayTimeout = 504


class SimilarityMetricType:
    l2 = "l2_norm"
    cosine = "cosine"
    dot = "dot_product"

    def get_attr(self, name):
        return getattr(self, name, "")


@dataclass
class ESParams:
    database_params: Optional[dict] = field(default_factory=lambda: {})
    insert_params: Optional[dict] = field(default_factory=lambda: {})
    connection_params: Optional[dict] = field(default_factory=lambda: {})
    indices_params: Optional[dict] = field(default_factory=lambda: {})
    force_merge_params: Optional[dict] = field(default_factory=lambda: {})
    search_params: Optional[dict] = field(default_factory=lambda: {})
    concurrent_params: Optional[dict] = field(default_factory=lambda: {})
    concurrent_tasks: Optional[list] = field(default_factory=lambda: [])


@dataclass
class ConcurrentTasksParams:
    type: int
    weight: Optional[int] = 0
    params: Optional[dict] = field(default_factory=lambda: {})
    other_params: Optional[dict] = field(default_factory=lambda: {})


@dataclass
class ConcurrentTasks:
    search: Optional[ConcurrentTasksParams] = field(default_factory=lambda: ConcurrentTasksParams(**{"type": "search"}))


@dataclass
class ESConcurrentParams:
    concurrent_during_time: int
    parallel: int
    interval: int
    warm_time: Optional[int] = 0

    # search params
    search_nq: Optional[int] = 0
    search_vectors_len: Optional[int] = 0
    search_params: Optional[dict] = field(default_factory=lambda: {})
    search_other_params: Optional[dict] = field(default_factory=lambda: {})


