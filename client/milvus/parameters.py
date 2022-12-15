import copy
import math
from typing import List, Optional
from dataclasses import dataclass, field
from common.common_func import read_file, milvus_gen_vectors
from client.base.parameters import ParametersBase
from client.milvus.define_params import (
    MILVUS_DEFAULT_FIELD_NAME,
    MilvusPrams,
    MILVUS_DEFAULT_COLLECTION,
    MILVUS_DEFAULT_METRIC_TYPE,
    MILVUS_DEFAULT_DIM
)


@dataclass
class ConcurrentTasksParams:
    type: int
    weight: Optional[int] = 0
    params: Optional[dict] = field(default_factory=lambda: {})
    other_params: Optional[dict] = field(default_factory=lambda: {})


@dataclass
class ConcurrentTasks:
    search: Optional[ConcurrentTasksParams] = field(default_factory=lambda: ConcurrentTasksParams(**{"type": "search"}))
    query: Optional[ConcurrentTasksParams] = field(default_factory=lambda: ConcurrentTasksParams(**{"type": "query"}))


class ParametersMilvus(ParametersBase):
    def __init__(self, params: dict):
        self.params = MilvusPrams(**params)
        self.update_default_params()

        self.concurrent_tasks_params = {}
        self.concurrent_tasks = None
        self.concurrent_tasks_parser()

    def update_default_params(self):
        if "collection_name" not in self.params.collection_params:
            self.params.collection_params["collection_name"] = MILVUS_DEFAULT_COLLECTION
        if "metric_type" not in self.params.database_params:
            self.params.database_params["metric_type"] = MILVUS_DEFAULT_METRIC_TYPE
        if "dim" not in self.params.database_params:
            self.params.database_params["dim"] = MILVUS_DEFAULT_DIM

    def concurrent_tasks_parser(self):
        p = copy.deepcopy(self.params.concurrent_tasks)
        t = {}
        for task in p:
            if task["type"] == "search":
                task["params"], task["other_params"] = self.search_params(task["params"], MILVUS_DEFAULT_FIELD_NAME,
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
            raise Exception("[CommonCases] Params of query are needed.")

        elif ids is not None:
            _expr = "id in %s" % str(ids)

        elif expr is not None:
            _expr = expr
        kwargs.update(expr=_expr)
        return kwargs

    def search_params(self, _search_params: dict, field_name: str, metric_type: str):
        _params = copy.deepcopy(_search_params)

        limit = _params.pop("top_k")
        search_param = _params.pop("search_param")
        expr = self.parser_search_params_expr(_params.pop("expr")) if "expr" in _params else None

        _params.update({
            "param": {"params": search_param, "metric_type": metric_type},
            "limit": limit,
            "expr": expr,
        })
        if "anns_field" not in _params:
            _params.update({"anns_field": field_name})

        search_vectors = _params.pop("search_vectors", None)
        nq = _params.pop("nq")
        if search_vectors:
            search_vectors = read_file(search_vectors)
        else:
            search_vectors = milvus_gen_vectors(nb=nq, dim=self.params.database_params["dim"])

        # generate vectors for recursive search
        vectors_len = len(search_vectors)
        lcm = self.least_common_multiple([nq, vectors_len])
        search_vectors = search_vectors * int((lcm / vectors_len))

        other_params = {
            "nq": nq,
            "search_vectors": search_vectors
        }
        return _params, other_params

    @staticmethod
    def compare_expr(left, comp, right):
        if comp == "LT":
            return "{0} < {1}".format(left, right)
        elif comp == "LE":
            return "{0} <= {1}".format(left, right)
        elif comp == "EQ":
            return "{0} == {1}".format(left, right)
        elif comp == "NE":
            return "{0} != {1}".format(left, right)
        elif comp == "GE":
            return "{0} >= {1}".format(left, right)
        elif comp == "GT":
            return "{0} > {1}".format(left, right)
        raise Exception("[compare_expr] Not support expr: {0}".format(comp))

    def parser_search_params_expr(self, expr):
        """
        :param expr:
            LT: less than
            LE: less than or equal to
            EQ: equal to
            NE: not equal to
            GE: greater than or equal to
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
                        _e = self.compare_expr(field_name, k, v)
                        expression = _e if expression == "" else "{0} && {1}".format(expression, _e)
        else:
            raise Exception(
                "[parser_search_params_expr] Can't parser search expression: {0}, type:{1}".format(expr, type(expr)))
        if expression == "":
            expression = None
        return expression

    @staticmethod
    def least_common_multiple(args: List[int]):
        def lcm(a: int, b: int):
            return int(a * b / math.gcd(a, b))

        if len(args) == 0:
            return 0
        elif len(args) == 1:
            return args[0]
        else:
            _lcm = args[0]
            for i in range(1, len(args)):
                _lcm = lcm(_lcm, args[i])
            return _lcm
