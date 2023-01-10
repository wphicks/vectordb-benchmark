import abc
import os
import threading
import multiprocessing
from utils.util_log import log


class ClientBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.p_obj = None  # object for parameters
        self.interval = 0
        self.warm_time = 0
        self.during_time = 0
        self.parallel = 0
        self.initializer = self.init_db
        self.init_args = ()
        self.pool_func = self.concurrent_pool_function
        self.iterable = iter([])

        # flag
        self.stop_concurrent_flag = False

        self.concurrent_params = None

    @abc.abstractmethod
    def init_db(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_concurrent_start_params(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_serial_start_params(self, *args, **kwargs):
        pass

    def concurrent_pool_function(self, params):
        api_type = params[0]
        params_func = params[1]
        result = []
        obj = eval(f"self.__class__.i_obj.{api_type}")
        log.debug(
            f" Start Concurrent API:{api_type}, PID:{os.getpid()}, {multiprocessing.current_process().name} ".center(
                100, '#'))
        self.concurrent_timer(self.warm_time * 2 + self.during_time, self.concurrent_stop)
        while not self.stop_concurrent_flag:
            result.append(obj(**next(params_func())))
        return result

    @staticmethod
    def concurrent_timer(during_time, func, args=None, kwargs=None):
        t = threading.Timer(during_time, func, args=args, kwargs=kwargs)
        t.start()

    def concurrent_stop(self, stop_flag=True):
        log.debug(f" Stop Concurrent PID:{os.getpid()}, {multiprocessing.current_process().name} ".center(100, '*'))
        self.stop_concurrent_flag = stop_flag
