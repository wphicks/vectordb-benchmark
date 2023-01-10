import copy
import numpy as np
from pprint import pformat
from datasets import DATASET_FILES_DIR
from common.common_func import read_search_file, milvus_gen_vectors, gen_combinations, normalize_data
from client.base.parameters import ParametersBase
from client.milvus.define_params import (
    MILVUS_DEFAULT_FIELD_NAME,
    MilvusParams,
    MILVUS_DEFAULT_COLLECTION,
    MILVUS_DEFAULT_METRIC_TYPE,
    MILVUS_DEFAULT_MAX_LENGTH,
    MILVUS_DEFAULT_DIM,
    ConcurrentTasksParams,
    ConcurrentTasks
)


class ParametersMilvus(ParametersBase):
    def __str__(self):
        return str(pformat(vars(self.params), sort_dicts=False))

    def __init__(self, params: dict):
        self.params = MilvusParams(**params)
        self.update_default_params()

        self.serial_params = None
        self.serial_search_params = []
        self.concurrent_tasks = None

    def update_default_params(self):
        if "collection_name" not in self.params.collection_params:
            self.params.collection_params["collection_name"] = MILVUS_DEFAULT_COLLECTION
        if "metric_type" not in self.params.database_params:
            self.params.database_params["metric_type"] = MILVUS_DEFAULT_METRIC_TYPE
        if "dim" not in self.params.database_params:
            self.params.database_params["dim"] = MILVUS_DEFAULT_DIM
        if "max_length" not in self.params.database_params:
            self.params.database_params["max_length"] = MILVUS_DEFAULT_MAX_LENGTH

    def reset_default_params(self, collection_name: str = None, metric_type: str = None, dim: int = None):
        self.params.collection_params["collection_name"] = collection_name or self.params.collection_params[
            "collection_name"]
        self.params.database_params["metric_type"] = metric_type or self.params.database_params["metric_type"]
        self.params.database_params["dim"] = dim or self.params.database_params["dim"]

    def serial_params_parser(self, collection_name: str = None, metric_type: str = None, dim: int = None):
        self.reset_default_params(collection_name, metric_type, dim)
        self.serial_params = copy.deepcopy(self.params)
        self.serial_params.index_params.update({"metric_type": self.params.database_params["metric_type"]})
        self.serial_params.collection_params.update({"dim": self.params.database_params["dim"]})
        self.serial_params.collection_params.update({"max_length": self.params.database_params["max_length"]})

        # parser search params to iter list
        serial_search_params = copy.deepcopy(self.params.search_params)
        if "search_param" in serial_search_params:
            serial_search_params["search_param"] = gen_combinations(serial_search_params["search_param"])
        s_p = gen_combinations({"top_k": serial_search_params.pop("top_k", 0),
                                "nq": serial_search_params.pop("nq", 0),
                                "search_param": serial_search_params.pop("search_param", {})})
        for s in s_p:
            s.update(serial_search_params)
            s.update({"metric_type": self.params.database_params["metric_type"]})
            self.serial_search_params.append(s)

    def concurrent_tasks_parser(self, metric_type: str = None, dim: int = None, anns_field=MILVUS_DEFAULT_FIELD_NAME):
        self.reset_default_params(metric_type=metric_type, dim=dim)
        p = copy.deepcopy(self.params.concurrent_tasks)
        t = {}
        for task in p:
            if task["type"] == "search":
                task["params"], task["other_params"] = self.search_params(task["params"], anns_field,
                                                                          self.params.database_params["metric_type"])
            elif task["type"] == "query":
                task["params"] = self.query_param_analysis(**task["params"])
                task["other_params"] = {}

            t.update({task["type"]: ConcurrentTasksParams(**{"type": task["type"],
                                                             "weight": task["weight"],
                                                             "params": task["params"],
                                                             "other_params": task["other_params"]})})
        self.concurrent_tasks = ConcurrentTasks(**t)

    @staticmethod
    def query_param_analysis(**kwargs):
        """
        :return: params for query
        """
        ids = kwargs.pop("ids", None)
        expr = kwargs.pop("expr", None)

        _expr = ""
        if ids is None and expr is None:
            raise Exception("[ParametersMilvus] Params of query are needed.")

        elif ids is not None:
            _expr = "id in %s" % str(ids)

        elif expr is not None:
            _expr = expr
        kwargs.update(expr=_expr)
        return kwargs

    def search_params(self, _search_params: dict, field_name: str = MILVUS_DEFAULT_FIELD_NAME,
                      metric_type: str = MILVUS_DEFAULT_METRIC_TYPE, serial=False, vectors=None):
        _params = copy.deepcopy(_search_params)

        limit = _params.pop("top_k")
        search_param = _params.pop("search_param")
        metric_type = _params.pop("metric_type", metric_type)
        expr = self.parser_search_params_expr(_params.pop("expr")) if "expr" in _params else None

        _params.update({
            "param": {"params": search_param, "metric_type": metric_type},
            "limit": limit,
            "expr": expr,
        })
        if "anns_field" not in _params:
            _params.update({"anns_field": field_name})
        nq = _params.pop("nq")

        if serial:
            _params.update({"data": vectors})
            return _params

        search_vectors_file = _params.pop("search_vectors", None)
        if search_vectors_file:
            search_vectors = read_search_file(search_vectors_file, DATASET_FILES_DIR)
            if str(search_vectors_file).endswith("hdf5"):
                search_vectors = normalize_data(metric_type, np.array(search_vectors))
        else:
            search_vectors = milvus_gen_vectors(nb=nq, dim=self.params.database_params["dim"])

        # generate vectors for recursive search
        vectors_len = len(search_vectors)
        lcm = self.least_common_multiple([nq, vectors_len])
        search_vectors = search_vectors * int(lcm / vectors_len)

        other_params = {
            "nq": nq,
            "search_vectors": search_vectors
        }
        return _params, other_params

    @staticmethod
    def compare_expr(left, comp, right):
        if comp == "LT":
            return "{0} < {1}".format(left, right)
        elif comp == "LTE":
            return "{0} <= {1}".format(left, right)
        elif comp == "EQ":
            return "{0} == {1}".format(left, right)
        elif comp == "NE":
            return "{0} != {1}".format(left, right)
        elif comp == "GTE":
            return "{0} >= {1}".format(left, right)
        elif comp == "GT":
            return "{0} > {1}".format(left, right)
        raise Exception("[ParametersMilvus] Not support expr: {0}".format(comp))

    def parser_search_params_expr(self, expr):
        """
        :param expr:
            LT: less than
            LTE: less than or equal to
            EQ: equal to
            NE: not equal to
            GTE: greater than or equal to
            GT: greater than
        :return: expression of search
        """
        if expr is None:
            return expr

        expression = ""
        if isinstance(expr, str):
            return expr
        elif isinstance(expr, dict):
            for key, value in expr.items():
                field_name = key
                if isinstance(value, dict):
                    for k, v in value.items():
                        _e = self.compare_expr(str(field_name).upper(), k, v)
                        expression = _e if expression == "" else "{0} && {1}".format(expression, _e)
        else:
            raise Exception(
                "[ParametersMilvus] Can't parser search expression: {0}, type:{1}".format(expr, type(expr)))
        if expression == "":
            expression = None
        return expression
