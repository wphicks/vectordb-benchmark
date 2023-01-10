import tqdm
import time
from client.base.client_base import ClientBase
from client.elasticsearch.interface import InterfaceElasticsearch
from client.elasticsearch.parameters import ParametersElasticsearch
from client.elasticsearch.define_params import ESConcurrentParams, DEFAULT_PRECISION, SimilarityMetricType
from datasets.reader import ReaderBase
from utils.util_log import log


class ClientElasticsearch(ClientBase):
    def __init__(self, params: dict, host: str = None, reader: ReaderBase = None):
        super().__init__()
        self.host = host
        self.params = params
        self.reader = reader

        self.p_obj = ParametersElasticsearch(params)
        self.i_obj = InterfaceElasticsearch(self.host)

        # flag
        self.start_subscript = 0

    def serial_prepare_data(self, prepare=True):
        log.info("[ClientElasticsearch] Start preparing data")
        self.i_obj.connect(self.host, **self.p_obj.params.connection_params)
        if not prepare:
            self.i_obj.connect_indices(self.p_obj.params.indices_params["index"])
        else:
            self.i_obj.clean_all_indices()
            self.i_obj.create_indices(**self.p_obj.serial_params.indices_params)

            # insert vectors
            log.info("[ClientElasticsearch] Start inserting data")
            insert_times = []
            for ids, vectors in tqdm.tqdm(self.reader.iter_train_vectors(self.p_obj.params.insert_params["batch"])):
                insert_times.append(self.i_obj.insert_batch(vectors, ids))
            insert_time = round(sum(insert_times), DEFAULT_PRECISION)

            log.info("[ClientElasticsearch] Waiting for forced merge index")
            index_start = time.perf_counter()
            self.i_obj.wait_index(**self.p_obj.params.force_merge_params)
            index_time = round(time.perf_counter() - index_start, DEFAULT_PRECISION)

            log.info(f"[ClientElasticsearch] Insert time:{insert_time}s, Wait index time:{index_time}s")
        log.info("[ClientElasticsearch] Data preparation completed")

    def serial_search_recall(self):
        for p in self.p_obj.serial_search_params:
            recall_list = []
            for s in tqdm.tqdm(self.reader.iter_test_vectors(p["nq"], p["top_k"])):
                search_params = self.p_obj.search_params(p, vectors=s.vectors[0], serial=True)
                recall_list.append(self.i_obj.search_recall(s.neighbors, **search_params))
            recall = round(sum(recall_list) / len(recall_list), DEFAULT_PRECISION)
            log.info(f"[ClientElasticsearch] Search recall:{recall}, search params:{p}")

    def get_serial_start_params(self, rb: ReaderBase):
        self.reader = rb
        metric_type = SimilarityMetricType().get_attr(rb.config.similarity_metric_type)
        self.p_obj.serial_params_parser(metric_type=metric_type, dim=rb.config.dim)
        log.info("[ClientElasticsearch] Parameters used: \n{}".format(self.p_obj))

    def get_concurrent_start_params(self):
        self.init_db()
        field_name, dim = self.__class__.i_obj.get_indices_params()

        self.p_obj.concurrent_tasks_parser(dim=dim, field_name=field_name)
        search = self.p_obj.concurrent_tasks.search
        self.concurrent_params = ESConcurrentParams(**{
            "concurrent_during_time": self.p_obj.params.concurrent_params["during_time"],
            "parallel": self.p_obj.params.concurrent_params["concurrent_number"],
            "interval": self.p_obj.params.concurrent_params["interval"],
            "warm_time": self.p_obj.params.concurrent_params[
                "warm_time"] if "warm_time" in self.p_obj.params.concurrent_params else 0,

            "search_nq": search.other_params.get("nq", 0),
            "search_vectors_len": len(search.other_params.get("search_vectors", 0)),
            "search_params": search.params,
            "search_other_params": search.other_params
        })

        iterable_params = []
        parallel = self.p_obj.params.concurrent_params["concurrent_number"]
        total_weights = search.weight
        search_parallel = round((search.weight / total_weights) * parallel)
        for s in range(search_parallel):
            iterable_params.append(("search", self.concurrent_search_iterable_params))

        self.interval = self.concurrent_params.interval
        self.parallel = self.concurrent_params.parallel
        self.warm_time = self.concurrent_params.warm_time
        self.during_time = self.concurrent_params.concurrent_during_time
        self.initializer = self.init_db
        self.init_args = ()
        self.pool_func = self.concurrent_pool_function
        self.iterable = iter(iterable_params)

    def init_db(self):
        self.__class__.i_obj = InterfaceElasticsearch(self.host)
        self.__class__.i_obj.connect(self.host, **self.p_obj.params.connection_params)
        self.__class__.i_obj.connect_indices(self.p_obj.params.indices_params["index"])

    def concurrent_search_iterable_params(self):
        while True:
            for p in range(self.parallel):
                end_subscript = self.start_subscript + self.concurrent_params.search_nq
                self.concurrent_params.search_params["knn"]["query_vector"] = \
                    self.concurrent_params.search_other_params["search_vectors"][self.start_subscript:end_subscript][0]
                self.start_subscript = end_subscript if end_subscript < self.concurrent_params.search_vectors_len else 0
                yield self.concurrent_params.search_params
