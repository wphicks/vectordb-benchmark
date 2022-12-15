import time
from pymilvus import DataType, DefaultConfig, CollectionSchema, FieldSchema, connections, Collection, utility
from client.base.interface import InterfaceBase
from utils.util_log import log
from client.milvus.define_params import (
    MILVUS_DEFAULT_DESCRIPTION,
    MILVUS_DEFAULT_MAX_LENGTH,
    MILVUS_DEFAULT_DIM,
    MILVUS_DEFAULT_FIELD_NAME,
    DEFAULT_PRECISION,
)


def milvus_catch():
    def wrapper(func):
        def inner_wrapper(*args, **kwargs) -> float:
            start = time.perf_counter()
            content = f"##['{func.__name__}', "
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

    def connect(self, host=DefaultConfig.DEFAULT_HOST, port=DefaultConfig.DEFAULT_PORT, **kwargs):
        self.connection = connections.connect(alias=DefaultConfig.DEFAULT_USING, host=host, port=port, **kwargs)

    def connect_collection(self, collection_name):
        self.collection = Collection(name=collection_name)

    def create_collection(self, collection_name="", vector_field_name=MILVUS_DEFAULT_FIELD_NAME, schema=None,
                          other_fields=[], shards_num=2, **kwargs):
        schema = schema or self.gen_collection_schema(vector_field_name=vector_field_name, other_fields=other_fields,
                                                      max_length=kwargs.get("max_length", MILVUS_DEFAULT_MAX_LENGTH),
                                                      dim=kwargs.get("dim", MILVUS_DEFAULT_DIM))
        self.collection = Collection(collection_name, schema=schema, shards_num=shards_num, **kwargs)

    @milvus_catch()
    def search(self, data, anns_field, param, limit, expr=None, timeout=300, **kwargs):
        """
        :return: rt
        example: 200
        """
        return self.collection.search(data, anns_field, param, limit, expr=expr, timeout=timeout, **kwargs)

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

    def gen_field_schema(self, name: str, dtype=None, description=MILVUS_DEFAULT_DESCRIPTION, is_primary=False, **kwargs):
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
