import threading
import os
import multiprocessing
from client.base.client_base import ClientBase
from client.milvus.interface import InterfaceMilvus
from client.milvus.parameters import ParametersMilvus
from client.milvus.define_params import MilvusConcurrentParams
from utils.util_log import log


class ClientMilvus(ClientBase):
    def __init__(self, params: dict, host: str = None):
        super().__init__()
        self.host = host
        self.params = params

        self.p_obj = ParametersMilvus(params)
        self.i_obj = InterfaceMilvus(self.host)

        # flag
        self.stop_concurrent_flag = False
        self.start_subscript = 0

        self.concurrent_params = None

    def get_concurrent_start_params(self):
        search = self.p_obj.concurrent_tasks.search
        query = self.p_obj.concurrent_tasks.query
        self.concurrent_params = MilvusConcurrentParams(**{
            "concurrent_during_time": self.p_obj.params.concurrent_params["during_time"],
            "interval": self.p_obj.params.concurrent_params["interval"],
            "parallel": self.p_obj.params.concurrent_params["concurrent_number"],
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
        self.concurrent_timer_stop(self.concurrent_params.concurrent_during_time, self.concurrent_stop)
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
    def concurrent_timer_stop(during_time, func, args=None, kwargs=None):
        t = threading.Timer(during_time, func, args=args, kwargs=kwargs)
        t.start()

    def concurrent_stop(self, stop_flag=True):
        log.debug(f" Stop Concurrent PID:{os.getpid()}, {multiprocessing.current_process().name} ".center(100, '*'))
        self.stop_concurrent_flag = stop_flag
