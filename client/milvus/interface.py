import time
import string
import random
import pandas as pd
from pymilvus import DataType, DefaultConfig, CollectionSchema, FieldSchema, connections, Collection, utility, Index
from client.base.interface import InterfaceBase
from utils.util_log import log
from client.milvus.define_params import (
    MILVUS_DEFAULT_DESCRIPTION,
    MILVUS_DEFAULT_MAX_LENGTH,
    MILVUS_DEFAULT_DIM,
    MILVUS_DEFAULT_FIELD_NAME,
    DEFAULT_PRECISION,
    MILVUS_DEFAULT_METRIC_TYPE,
)


def milvus_catch():
    def wrapper(func):
        def inner_wrapper(*args, **kwargs) -> float:
            content = f"##['{func.__name__}', "
            start = time.perf_counter()
            try:
                res = func(*args, **kwargs)
                rt = (time.perf_counter() - start) * 1000
                return_res = (True, round(rt, DEFAULT_PRECISION), res)
                content += "{0}, {1}]## {2}".format(*return_res)
                return rt
            except Exception as e:
                rt = (time.perf_counter() - start) * 1000

                return_res = (False, round(rt, DEFAULT_PRECISION), e)
                content += "{0}, {1}]## {2}".format(*return_res)
                return rt
            finally:
                log.debug(content)
        return inner_wrapper
    return wrapper


class InterfaceMilvus(InterfaceBase):
    def __init__(self, host=DefaultConfig.DEFAULT_HOST, port=DefaultConfig.DEFAULT_PORT):
        self.host = host or DefaultConfig.DEFAULT_HOST
        self.port = port or DefaultConfig.DEFAULT_PORT

        self.connection = None
        self.collection = None
        self.collection_schema = None

    def connect(self, host=DefaultConfig.DEFAULT_HOST, port=DefaultConfig.DEFAULT_PORT, **kwargs):
        self.connection = connections.connect(alias=DefaultConfig.DEFAULT_USING, host=host, port=port, **kwargs)

    def connect_collection(self, collection_name):
        self.collection = Collection(name=collection_name)

    @staticmethod
    def clean_all_collection():
        collections = utility.list_collections()
        log.debug("[InterfaceMilvus] Start cleaning all collections: {}".format(collections))
        for i in collections:
            utility.drop_collection(i)

    def get_collection_params(self):
        field_name = MILVUS_DEFAULT_FIELD_NAME
        dim = MILVUS_DEFAULT_DIM
        metric_type = MILVUS_DEFAULT_METRIC_TYPE

        for field in self.collection.schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                field_name = field.name
                dim = field.params.get("dim")
        if self.collection.has_index():
            metric_type = self.collection.index().params.get("metric_type")

        return field_name, dim, metric_type

    def create_collection(self, collection_name="", vector_field_name=MILVUS_DEFAULT_FIELD_NAME, schema=None,
                          other_fields=[], shards_num=2, **kwargs):
        schema = schema or self.gen_collection_schema(vector_field_name=vector_field_name, other_fields=other_fields,
                                                      max_length=kwargs.get("max_length", MILVUS_DEFAULT_MAX_LENGTH),
                                                      dim=kwargs.get("dim", MILVUS_DEFAULT_DIM))
        log.debug("[InterfaceMilvus] Start create collection: {}".format(collection_name))
        self.collection = Collection(collection_name, schema=schema, shards_num=shards_num, **kwargs)

    def insert_batch(self, vectors, ids, varchar_filled=False):
        """
        :return: rt / s
        """
        if self.collection_schema is None:
            self.collection_schema = self.collection.schema.to_dict()

        if isinstance(vectors, list):
            entities = self.gen_entities(self.collection_schema, vectors, ids, varchar_filled)
        else:
            entities = self.gen_entities(self.collection_schema, vectors.tolist(), ids, varchar_filled)

        start = time.perf_counter()
        res = self.collection.insert(entities)
        return round(time.perf_counter() - start, DEFAULT_PRECISION)

    def flush_collection(self):
        if not hasattr(self.collection, "flush"):
            return self.collection.num_entities
        return self.collection.flush()

    def wait_for_compaction_completed(self):
        self.collection.compact()
        self.collection.wait_for_compaction_completed()

    def load_collection(self, **kwargs):
        log.debug("[InterfaceMilvus] Start loading.")
        return self.collection.load(**kwargs)

    def build_index(self, index_type, metric_type, index_param, field_name=MILVUS_DEFAULT_FIELD_NAME):
        """
        {"index_type": "IVF_FLAT", "metric_type": "L2", "index_param": {"nlist": 128}}
        """
        log.debug("[InterfaceMilvus] Start building index: {}".format(index_type))
        index_params = {"index_type": index_type, "metric_type": metric_type, "params": index_param}
        return self.collection.create_index(field_name, index_params)

    @milvus_catch()
    def search(self, data, anns_field, param, limit, expr=None, timeout=300, **kwargs):
        """
        :return: rt / ms
        example: 200
        """
        return self.collection.search(data, anns_field, param, limit, expr=expr, timeout=timeout, **kwargs)

    def search_recall(self, true_ids, data, anns_field, param, limit, expr=None, timeout=300, **kwargs):
        """
        :return: recall
        example: 1.0
        """
        res = self.collection.search(data, anns_field, param, limit, expr=expr, timeout=timeout, **kwargs)
        ids = self.get_search_ids(res)
        recall = self.get_recall_value(true_ids=true_ids, result_ids=ids)
        return recall

    @milvus_catch()
    def query(self, **kwargs):
        """
        :return: rt
        example: 200
        """
        return self.collection.query(**kwargs)

    @staticmethod
    def clean_all_collections():
        for c in utility.list_collections():
            utility.drop_collection(c)

    @staticmethod
    def list_all_collections():
        return utility.list_collections()

    @staticmethod
    def field_type():
        """
        'bool', 'int8', 'int16', 'int32', 'int64', 'float', 'double',
        'string', 'varchar', 'binary_vector', 'float_vector'
        """
        data_types = dict(DataType.__members__)
        _field_types = {}
        for i in data_types.keys():
            if str(i) not in ["NONE", "UNKNOWN"]:
                _field_types.update(i=data_types[i])
        data_types = dict(sorted(data_types.items(), key=lambda item: item[0], reverse=True))
        return data_types

    def gen_field_schema(self, name: str, dtype=None, description=MILVUS_DEFAULT_DESCRIPTION, is_primary=False,
                         **kwargs):
        field_types = self.field_type()
        if dtype is None:
            for _field in field_types.keys():
                if name.startswith(_field.lower()):
                    _kwargs = {}
                    if _field in ["STRING", "VARCHAR"]:
                        _kwargs.update({"max_length": kwargs.get("max_length", MILVUS_DEFAULT_MAX_LENGTH)})
                    if _field in ["BINARY_VECTOR", "FLOAT_VECTOR"]:
                        _kwargs.update({"dim": kwargs.get("dim", MILVUS_DEFAULT_DIM)})
                    return FieldSchema(name=name, dtype=field_types[_field], description=description,
                                       is_primary=is_primary, **_kwargs)
        else:
            if dtype in field_types.values():
                return FieldSchema(name=name, dtype=dtype, description=description, is_primary=is_primary, **kwargs)
        return []

    def gen_collection_schema(self, vector_field_name="", description=MILVUS_DEFAULT_DESCRIPTION, default_fields=True,
                              auto_id=False, other_fields=[], primary_field=None, **kwargs):
        fields = [self.gen_field_schema("id", dtype=DataType.INT64, is_primary=True),
                  self.gen_field_schema(vector_field_name,
                                        dim=kwargs.get("dim", MILVUS_DEFAULT_DIM))] if default_fields else []

        for _field in other_fields:
            fields.append(self.gen_field_schema(_field, **kwargs))
        return CollectionSchema(fields=fields, description=description, auto_id=auto_id, primary_field=primary_field)

    @staticmethod
    def gen_values(data_type, vectors, ids, varchar_filled=False, field={}):
        values = None
        if data_type in [DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64]:
            values = ids
        elif data_type in [DataType.DOUBLE]:
            values = [(i + 0.0) for i in ids]
        elif data_type == DataType.FLOAT:
            values = pd.Series(data=[(i + 0.0) for i in ids], dtype="float32")
        elif data_type in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
            values = vectors
        elif data_type in [DataType.VARCHAR]:
            if varchar_filled is False:
                values = [str(i) for i in ids]
            else:
                _len = int(field["params"]["max_length"])
                _str = string.ascii_letters + string.digits
                _s = _str
                for i in range(int(_len / len(_str))):
                    _s += _str
                values = [''.join(random.sample(_s, _len - 1)) for i in ids]
        return values

    def gen_entities(self, info, vectors=None, ids=None, varchar_filled=False):
        if not isinstance(info, dict):
            raise Exception("[InterfaceMilvus] info is not a dict, please check: {}".format(type(info)))
        if "fields" not in info:
            raise Exception("[InterfaceMilvus] fields not in info, please check: {}".format(info))

        entities = {}
        for field in info["fields"]:
            _type = field["type"]
            entities.update({field["name"]: self.gen_values(_type, vectors, ids, varchar_filled, field)})
        return pd.DataFrame(entities)

    @staticmethod
    def get_search_ids(result):
        ids = []
        for res in result:
            ids.append(res.ids)
        return ids
