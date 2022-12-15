import threading
import time
from utils.util_log import log


class StreamRead:
    def __init__(self, file_path: str, interval: int = 600, stop_read_flag: bool = False):
        self.file_path = file_path
        self.interval = interval
        self.stop_read_flag = stop_read_flag
        self.read_finished = False
        self.tick_read_flag = True

        self._streaming_read_incremental_file = self.streaming_read_incremental_file()
        self.t = None  # threading object

    def set_stop_read_flag(self, stop_flag: bool = True):
        """ Set the flag to True to stop streaming reading """
        self.stop_read_flag = stop_flag

    def set_read_finished(self, stop_flag: bool = True):
        """ Set the flag to True to stop reading """
        self.read_finished = stop_flag

    def streaming_read_incremental_file(self, file_path: str = ""):
        file_path = file_path or self.file_path

        with open(file_path) as fd:

            last_position = 0
            while True:
                incremental_content = fd.read()

                current_position = fd.tell()
                if current_position != last_position:
                    fd.seek(current_position, 0)
                last_position = current_position

                yield incremental_content

    def tick_read_incremental_file(self, callable_object: callable = print):
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
        incremental_content = next(self._streaming_read_incremental_file)
        callable_object(incremental_content)
        self.tick_read_flag = True

        if not self.stop_read_flag:
            self.t = threading.Timer(self.interval, self.tick_read_incremental_file, args=[callable_object])
            self.t.start()

    def final_read_incremental_file(self, callable_object: callable = print):
        self.set_stop_read_flag()
        start = time.time()
        while time.time() - start < self.interval:
            if self.tick_read_flag:
                if self.t:
                    self.t.cancel()
                incremental_content = next(self._streaming_read_incremental_file)
                callable_object(incremental_content)
                break
            log.debug("[StreamRead] Wait for the last read to complete.")
            time.sleep(2)
