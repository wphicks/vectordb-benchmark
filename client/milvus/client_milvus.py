import threading
import os
import multiprocessing
import numpy as np
import tqdm
import time
from client.base.client_base import ClientBase
from client.milvus.interface import InterfaceMilvus
from client.milvus.parameters import ParametersMilvus
from client.milvus.define_params import MilvusConcurrentParams, DEFAULT_PRECISION
from common.common_func import normalize_data
from datasets.reader import ReaderBase
from utils.util_log import log


class ClientMilvus(ClientBase):
    def __init__(self, params: dict, host: str = None, reader: ReaderBase = None):
        super().__init__()
        self.host = host
        self.params = params
        self.reader = reader

        self.p_obj = ParametersMilvus(params)
        self.i_obj = InterfaceMilvus(self.host)

        # flag
        self.stop_concurrent_flag = False
        self.start_subscript = 0

        self.concurrent_params = None

    def serial_prepare_data(self, prepare=True):
        self.i_obj.connect(self.host, **self.p_obj.params.connection_params)
        if not prepare:
            self.i_obj.connect_collection(self.p_obj.params.collection_params["collection_name"])
        else:
            self.i_obj.clean_all_collection()
            self.i_obj.create_collection(**self.p_obj.serial_params.collection_params)

            # insert vectors
            insert_start = time.perf_counter()
            for ids, vectors in tqdm.tqdm(self.reader.iter_train_vectors(self.p_obj.params.insert_params["batch"])):
                self.i_obj.insert_batch(vectors, ids)
            insert_time = round(time.perf_counter() - insert_start, DEFAULT_PRECISION)

            self.i_obj.flush_collection()
            index_start = time.perf_counter()
            self.i_obj.build_index(**self.p_obj.serial_params.index_params)
            index_time = round(time.perf_counter() - index_start, DEFAULT_PRECISION)
            # self.i_obj.wait_for_compaction_completed()
            # self.i_obj.build_index(**self.p_obj.serial_params.index_params)

            load_start = time.perf_counter()
            self.i_obj.load_collection(**self.p_obj.serial_params.load_params)
            load_time = round(time.perf_counter() - load_start, DEFAULT_PRECISION)
            log.info(f"[ClientMilvus] Insert time:{insert_time}s, Index time:{index_time}s, Load time:{load_time}s")

    def serial_search_recall(self):
        for p in self.p_obj.serial_search_params:
            recall_list = []
            for s in self.reader.iter_test_vectors(p["nq"], p["top_k"]):
                search_params = self.p_obj.search_params(p, vectors=s.vectors, serial=True)
                recall_list.append(self.i_obj.search_recall(s.neighbors, **search_params))
            recall = round(sum(recall_list) / len(recall_list), DEFAULT_PRECISION)
            log.info(f"[ClientMilvus] Search recall:{recall}, search params:{p}")

    def get_serial_start_params(self, rb: ReaderBase):
        self.reader = rb
        self.p_obj.serial_params_parser(metric_type=rb.config.similarity_metric_type, dim=rb.config.dim)
        rb.dataset_content.test = normalize_data(rb.config.similarity_metric_type, np.array(rb.dataset_content.test))
        rb.dataset_content.train = normalize_data(rb.config.similarity_metric_type, np.array(rb.dataset_content.train))
        log.debug("[ClientMilvus] Parameters used: \n{}".format(self.p_obj))

    def get_concurrent_start_params(self):
        self.init_db()
        field_name, dim, metric_type = self.__class__.i_obj.get_collection_params()

        self.p_obj.concurrent_tasks_parser(metric_type=metric_type, dim=dim, anns_field=field_name)
        search = self.p_obj.concurrent_tasks.search
        query = self.p_obj.concurrent_tasks.query
        self.concurrent_params = MilvusConcurrentParams(**{
            "concurrent_during_time": self.p_obj.params.concurrent_params["during_time"],
            "parallel": self.p_obj.params.concurrent_params["concurrent_number"],
            "interval": self.p_obj.params.concurrent_params["interval"],
            "warm_time": self.p_obj.params.concurrent_params[
                "warm_time"] if "warm_time" in self.p_obj.params.concurrent_params else 0,
            "search_nq": search.other_params.get("nq", 0),
            "search_vectors_len": len(search.other_params.get("search_vectors", 0)),
            "search_params": search.params,
            "search_other_params": search.other_params,
            "query_params": query.params,
            "query_other_params": query.other_params
        })

        iterable_params = []
        parallel = self.p_obj.params.concurrent_params["concurrent_number"]
        total_weights = search.weight + query.weight
        search_parallel = round((search.weight / total_weights) * parallel)
        query_parallel = round((query.weight / total_weights) * parallel)
        for s in range(search_parallel):
            iterable_params.append(("search", self.concurrent_search_iterable_params))
        for q in range(query_parallel):
            iterable_params.append(("query", self.concurrent_query_iterable_params))

        self.interval = self.concurrent_params.interval
        self.parallel = self.concurrent_params.parallel
        self.warm_time = self.concurrent_params.warm_time
        self.during_time = self.concurrent_params.concurrent_during_time
        self.initializer = self.init_db
        self.init_args = ()
        self.pool_func = self.concurrent_pool_function
        self.iterable = iter(iterable_params)

    def init_db(self):
        self.__class__.i_obj = InterfaceMilvus(self.host)
        self.__class__.i_obj.connect(self.host, **self.p_obj.params.connection_params)
        self.__class__.i_obj.connect_collection(self.p_obj.params.collection_params["collection_name"])

    def concurrent_pool_function(self, params):
        api_type = params[0]
        params_func = params[1]
        result = []
        obj = eval(f"self.__class__.i_obj.{api_type}")
        log.debug(
            f" Start Concurrent API:{api_type}, PID:{os.getpid()}, {multiprocessing.current_process().name} ".center(
                100, '#'))
        self.concurrent_timer(self.concurrent_params.warm_time * 2 + self.concurrent_params.concurrent_during_time,
                              self.concurrent_stop)
        while not self.stop_concurrent_flag:
            result.append(obj(**next(params_func())))
        return result

    def concurrent_query_iterable_params(self):
        while True:
            yield self.concurrent_params.query_params

    def concurrent_search_iterable_params(self):
        while True:
            for p in range(self.parallel):
                end_subscript = self.start_subscript + self.concurrent_params.search_nq
                self.concurrent_params.search_params["data"] = self.concurrent_params.search_other_params[
                                                                   "search_vectors"][self.start_subscript:end_subscript]
                self.start_subscript = end_subscript if end_subscript < self.concurrent_params.search_vectors_len else 0
                yield self.concurrent_params.search_params

    @staticmethod
    def concurrent_timer(during_time, func, args=None, kwargs=None):
        t = threading.Timer(during_time, func, args=args, kwargs=kwargs)
        t.start()

    def concurrent_stop(self, stop_flag=True):
        log.debug(f" Stop Concurrent PID:{os.getpid()}, {multiprocessing.current_process().name} ".center(100, '*'))
        self.stop_concurrent_flag = stop_flag
