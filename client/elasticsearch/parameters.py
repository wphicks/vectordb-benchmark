import copy
import numpy as np
from pprint import pformat
from datasets import DATASET_FILES_DIR
from common.common_func import read_search_file, milvus_gen_vectors, gen_combinations
from client.base.parameters import ParametersBase
from client.elasticsearch.define_params import (
    ES_DEFAULT_INDEX,
    ES_DEFAULT_FIELD_NAME,
    ES_DEFAULT_METRIC_TYPE,
    ES_DEFAULT_NUM_CANDIDATES,
    ES_DEFAULT_DIM,
    ESParams,
    ConcurrentTasksParams,
    ConcurrentTasks,
)


class ParametersElasticsearch(ParametersBase):
    def __str__(self):
        return str(pformat(vars(self.params), sort_dicts=False))

    def __init__(self, params: dict):
        self.params = ESParams(**params)
        self.update_default_params()

        self.serial_params = None
        self.serial_search_params = []
        self.concurrent_tasks = None

    def update_default_params(self):
        if "index" not in self.params.indices_params:
            self.params.indices_params["index"] = ES_DEFAULT_INDEX
        if "dim" not in self.params.database_params:
            self.params.database_params["dim"] = ES_DEFAULT_DIM

    def reset_default_params(self, index: str = None, dim: int = None, metric_type: str = None):
        self.params.indices_params["index"] = index or self.params.indices_params["index"]
        self.params.database_params["dim"] = dim or self.params.database_params["dim"]
        if metric_type:
            self.params.indices_params["metric_type"] = metric_type

    def serial_params_parser(self, index: str = None, dim: int = None, metric_type: str = ES_DEFAULT_METRIC_TYPE):
        self.reset_default_params(index, dim, metric_type)
        self.serial_params = copy.deepcopy(self.params)
        self.serial_params.indices_params.update({"dim": self.params.database_params["dim"]})

        # parser search params to iter list
        serial_search_params = copy.deepcopy(self.params.search_params)
        s_p = gen_combinations({"top_k": serial_search_params.pop("top_k", 0),
                                "nq": serial_search_params.pop("nq", 0),
                                "num_candidates": serial_search_params.pop("num_candidates",
                                                                           ES_DEFAULT_NUM_CANDIDATES)})
        for s in s_p:
            s.update(serial_search_params)
            self.serial_search_params.append(s)

    def concurrent_tasks_parser(self, dim: int = None, field_name=ES_DEFAULT_FIELD_NAME):
        self.reset_default_params(dim=dim)
        p = copy.deepcopy(self.params.concurrent_tasks)
        t = {}
        for task in p:
            if task["type"] == "search":
                task["params"], task["other_params"] = self.search_params(task["params"], field_name)

            t.update({task["type"]: ConcurrentTasksParams(**{"type": task["type"],
                                                             "weight": task["weight"],
                                                             "params": task["params"],
                                                             "other_params": task["other_params"]})})
        self.concurrent_tasks = ConcurrentTasks(**t)

    def search_params(self, _search_params: dict, field_name: str = ES_DEFAULT_FIELD_NAME, serial=False, vectors=None):
        s_p = copy.deepcopy(_search_params)
        top_k = s_p.pop("top_k")
        nq = s_p.pop("nq")

        knn = {
            "field": s_p.pop("field_name", field_name),
            "k": top_k,
            "num_candidates": s_p.pop("num_candidates", ES_DEFAULT_NUM_CANDIDATES),
            "filter": s_p.pop("filter", []),
        }
        if serial:
            knn.update({"query_vector": vectors})
            return {"knn": knn, "size": top_k, **s_p}

        search_vectors_file = s_p.pop("search_vectors", None)
        if search_vectors_file:
            search_vectors = read_search_file(search_vectors_file, DATASET_FILES_DIR)
            if str(search_vectors_file).endswith("hdf5"):
                search_vectors = np.array(search_vectors)
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
        return {"knn": knn, "size": top_k, **s_p}, other_params
