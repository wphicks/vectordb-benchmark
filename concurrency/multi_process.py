import time
import numpy as np
from pprint import pformat
import multiprocessing as mp
from multiprocessing import get_context

from concurrency.parser_result import ParserResult
from utils.util_log import log

DEFAULT_PRECISION = 4


class MultiProcessConcurrent:
    def __init__(self):
        pass

    @staticmethod
    def get_mp_start_method():
        mp_start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        log.info("[MultiProcessConcurrent] Get multiprocessing start method: {}".format(mp_start_method))
        return mp_start_method

    @staticmethod
    def wait_time(parallel):
        """ Wait for processes ready """
        t = min(parallel / 5, 5)
        log.info("[MultiProcessConcurrent] Start waiting for {0} processes ready: {1}s".format(parallel, t))
        time.sleep(t)

    def start(self, obj):
        log.info("[MultiProcessConcurrent] Parameters used: \n{0}".format(obj.p_obj))
        ctx = get_context(self.get_mp_start_method())
        result = []
        parser_result = ParserResult(interval=obj.interval)

        log.info("[MultiProcessConcurrent] Start initializing the concurrent pool")
        pool = None
        try:
            pool = ctx.Pool(processes=obj.parallel, initializer=obj.initializer, initargs=obj.init_args)
            self.wait_time(obj.parallel)
            log.info("[MultiProcessConcurrent] Start concurrent pool")
            parser_result.start_stream_read()

            start = time.perf_counter()
            for r in pool.imap_unordered(obj.pool_func, iterable=obj.iterable):
                result.extend(r)
            total_time = round(time.perf_counter() - start, DEFAULT_PRECISION)

            log.info("[MultiProcessConcurrent] End concurrent pool")
            parser_result.finish_stream_read(during_time=total_time)
        except Exception as e:
            raise Exception("[MultiProcessConcurrent] Concurrent pool failed: {}".format(e))
        finally:
            if pool:
                pool.close()
                pool.join()

        concurrent_result = {
            "reqs": len(result),
            "rps": round(len(result) / total_time, DEFAULT_PRECISION),
            "total_time_s": total_time,
            "avg_time_ms": round(np.mean(result), DEFAULT_PRECISION),
            "min_time_ms": round(np.min(result), DEFAULT_PRECISION),
            "max_time_ms": round(np.max(result), DEFAULT_PRECISION),
            "median_time_ms": round(np.median(result), DEFAULT_PRECISION),
            "p95_time_ms": round(np.percentile(result, 95), DEFAULT_PRECISION),
            "p99_time_ms": round(np.percentile(result, 99), DEFAULT_PRECISION),
            "sum_all_rt_ms": round(sum(result), DEFAULT_PRECISION)
        } if len(result) > 0 else {}
        log.info("[MultiProcessConcurrent] Summary of overall results: \n{0}".format(
            pformat(concurrent_result, sort_dicts=False)))

        return concurrent_result
