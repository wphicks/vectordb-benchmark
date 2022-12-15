import threading
import time
from datetime import datetime
from utils.util_log import log
from common.common_func import modify_file, write_existed_file
from results import LOG_FILE_DEBUG
import os


class ThreadTimer:
    def __init__(self, interval: int = 20, result_file: str = LOG_FILE_DEBUG):
        self.interval = interval
        self.result = []
        self.stop_read_flag = False
        self.tick_read_flag = True
        self.result_file = result_file

    def set_stop_read_flag(self, stop_flag: bool = True):
        """ Set the flag to True to stop streaming reading """
        self.stop_read_flag = stop_flag

    def get_incremental_result(self):
        result_len = len(self.result)
        incremental_content = self.result[:result_len]

        content = list(zip(*incremental_content))
        success = 0 if len(content) == 0 else sum(list(content[0]))
        failed = result_len - success

        content = str(result_len) + '-' + str(success) + '-' + str(failed) + '-' + str(
            content) + '-' + str(os.getpid())
        del self.result[:result_len]
        return content

    def tick_read_incremental_result(self, callable_object: callable = print):
        if self.stop_read_flag:
            return True

        run_count = 1
        while self.tick_read_flag is False:
            log.warning("[StreamRead] The time interval({0}s) is too short to complete a stream read, count:{1}".format(
                self.interval, run_count))
            time.sleep(self.interval)
            run_count += 1
            if run_count > 10:
                log.error("[StreamRead] The thread of streaming read may be stuck, please check")
                return False

        self.tick_read_flag = False
        callable_object(self.result_file, self.get_incremental_result())
        self.tick_read_flag = True

        if not self.stop_read_flag:
            t = threading.Timer(self.interval, self.tick_read_incremental_result, args=[callable_object])
            t.start()

    def final_read_incremental_result(self, callable_object: callable = modify_file):
        self.set_stop_read_flag()
        start = time.time()
        while time.time() - start < self.interval:
            if self.tick_read_flag:
                callable_object(self.result_file, self.get_incremental_result())
                break
            log.debug("[StreamRead] Wait for the last read to complete.")
            time.sleep(2)
