import string
import random
import time
from elasticsearch import Elasticsearch
from client.base.interface import InterfaceBase
from common.common_func import gen_zips
from utils.util_log import log
from client.elasticsearch.define_params import (
    ES_USER,
    ES_PASSWORD,
    ES_DEFAULT_HOST,
    ES_DEFAULT_PORT,
    ES_DEFAULT_INDEX,
    ES_DEFAULT_FIELD_NAME,
    ES_DEFAULT_INDEX_OPTIONS,
    ES_DEFAULT_METRIC_TYPE,
    ES_DEFAULT_TEXT_LENGTH,
    ES_DEFAULT_DIM,
    DEFAULT_PRECISION,
    ES_DEFAULT_TYPE,
    IndexSupport,
    INDEX_TYPE_MAPPING,
    INDEX_OPTIONS_MAPPING,
    HttpCode,
)


def es_catch():
    def wrapper(func):
        def inner_wrapper(*args, **kwargs) -> float:
            content = f"##['{func.__name__}', "
            result = False
            start = time.perf_counter()
            try:
                res = func(*args, **kwargs)
                rt = (time.perf_counter() - start) * 1000
                if res and res.meta.status == HttpCode.OK:
                    result = True
                    # rt = res.meta.duration * 1000
                return_res = (result, round(rt, DEFAULT_PRECISION), res)
                content += "{0}, {1}]## {2}".format(*return_res)
                return rt
            except Exception as e:
                rt = (time.perf_counter() - start) * 1000

                return_res = (result, round(rt, DEFAULT_PRECISION), e)
                content += "{0}, {1}]## {2}".format(*return_res)
                return rt
            finally:
                log.debug(content)

        return inner_wrapper

    return wrapper


class InterfaceElasticsearch(InterfaceBase):
    client: Elasticsearch = None

    def __init__(self, host=ES_DEFAULT_HOST, port=ES_DEFAULT_PORT):
        self.host = host or ES_DEFAULT_HOST
        self.port = port or ES_DEFAULT_PORT

        self.hosts = f"http://{self.host}:{self.port}"
        self.index = ES_DEFAULT_INDEX

    def connect(self, host="", **kwargs):
        self.host = host or self.host
        self.client = Elasticsearch(hosts=self.hosts, basic_auth=(ES_USER, ES_PASSWORD), verify_certs=False,
                                    retry_on_timeout=True, request_timeout=120, **kwargs)

    def clean_all_indices(self, timeout="2m"):
        try:
            indices = list(self.client.indices.get_mapping().keys())
            log.debug("[InterfaceElasticsearch] Start cleaning all indices: {}".format(indices))
            for i in indices:
                self.client.indices.delete(index=i, ignore_unavailable=True, timeout=timeout)
        except Exception as e:
            log.debug("[InterfaceElasticsearch] Can not get indices: {}".format(e))

    def connect_indices(self, index=ES_DEFAULT_INDEX):
        self.index = index

    def create_indices(self, index=ES_DEFAULT_INDEX, field_name=ES_DEFAULT_FIELD_NAME,
                       metric_type=ES_DEFAULT_METRIC_TYPE, timeout="2m", index_options=ES_DEFAULT_INDEX_OPTIONS,
                       other_fields=[], **kwargs):
        self.index = index
        other_fields_properties = {o: {"type": INDEX_TYPE_MAPPING.get(o, ES_DEFAULT_TYPE),
                                       **INDEX_OPTIONS_MAPPING.get(o, {})} for o in other_fields}
        mappings = {
            "properties": {
                field_name: {
                    "type": "dense_vector",
                    "dims": kwargs.pop("dim", ES_DEFAULT_DIM),
                    "index": True,
                    "similarity": metric_type,
                    "index_options": index_options,
                },
                **other_fields_properties,
            }
        }
        log.debug(
            f"[InterfaceElasticsearch] Create indices:{self.index}, mappings:{mappings}, timeout:{timeout}, kwargs:{kwargs}")
        return self.client.indices.create(index=self.index, mappings=mappings, timeout=timeout, **kwargs)

    def insert_batch(self, vectors, ids, timeout="10m"):
        """
        :return: rt / s
        """
        body = []
        vectors = vectors if isinstance(vectors, list) else vectors.tolist()
        for _id, vector in zip(ids, self.gen_entities(info=self.get_indices_info(), vectors=vectors, ids=ids)):
            body.append({"index": {"_id": _id}})
            body.append(vector)
        res = self.client.bulk(index=self.index, operations=body, timeout=timeout)
        return res.meta.duration

    @es_catch()
    def search(self, knn: dict, size: int, **kwargs):
        """
        :return: rt / ms
        example: 200
        """
        return self.client.search(index=self.index, knn=knn, size=size, **kwargs)

    def search_recall(self, true_ids, knn: dict, size: int, **kwargs):
        """
        :return: recall
        example: 1.0
        """
        res = self.client.search(index=self.index, knn=knn, size=size, **kwargs)
        ids = self.get_search_ids(res)
        recall = self.get_recall_value(true_ids=true_ids, result_ids=[ids])
        return recall

    def wait_index(self, **kwargs):
        return self.client.indices.forcemerge(index=self.index, wait_for_completion=True, **kwargs)

    def get_indices_info(self):
        return self.client.indices.get(index=self.index).body[self.index]

    def get_indices_params(self):
        field_name = ES_DEFAULT_FIELD_NAME
        dim = ES_DEFAULT_DIM

        for k, v in self.get_indices_info()["mappings"]["properties"].items():
            if v["type"] == IndexSupport.DenseVector:
                field_name = k
                dim = v["dims"]
        return field_name, dim

    def gen_entities(self, info, vectors=None, ids=None):
        if not isinstance(info, dict):
            raise Exception(
                "[InterfaceElasticsearch] info is not a dict, type:{0}, content:{1}".format(type(info), info))
        if "mappings" not in info and "properties" not in info["mappings"]:
            raise Exception(
                "[InterfaceElasticsearch] mappings not in info or properties not in info['mappings']: {}".format(info))

        entities = {}
        for k, v in info["mappings"]["properties"].items():
            entities.update({k: self.gen_values(k, v, vectors, ids)})
        return gen_zips(entities)

    @staticmethod
    def gen_values(properties, value, vectors, ids, text_length: int = ES_DEFAULT_TEXT_LENGTH):
        support_type = INDEX_TYPE_MAPPING.values()
        data_type = value.get("type", "")
        if data_type not in support_type:
            raise Exception(
                f"[InterfaceElasticsearch] Properties '{properties}':{value} do not yet support auto-generated data")

        values = None
        if data_type in [IndexSupport.Long, IndexSupport.Integer, IndexSupport.Short, IndexSupport.Byte]:
            values = ids
        elif data_type in [IndexSupport.Float, IndexSupport.Double]:
            values = [(i + 0.0) for i in ids]
        elif data_type in [IndexSupport.DenseVector]:
            values = vectors
        elif data_type in [IndexSupport.Text]:
            _str = string.ascii_letters + string.digits
            _s = _str
            for i in range(int(text_length / len(_str))):
                _s += _str
            values = [''.join(random.sample(_s, text_length - 1)) for i in ids]
        return values

    @staticmethod
    def get_search_ids(result):
        ids = []
        if "hits" not in result or "hits" not in result["hits"]:
            log.error("[InterfaceElasticsearch] Can not get search ids:{}".format(result))
            return ids
        return [int(res["_id"]) for res in result["hits"]["hits"]]
