import re
from datetime import datetime
from concurrency.streaming_read import StreamRead
from concurrency.data_client import DataParams, DataClient
from results import LOG_FILE_DEBUG
from utils.util_log import log


class ParserResult:
    def __init__(self, file_path=LOG_FILE_DEBUG, interval=20, warm_time=0, during_time=0):
        self.interval = interval
        self.warm_time = warm_time
        self.real_time = None
        self._format = self.data_parser_format()

        self.read_client = StreamRead(file_path=file_path, interval=interval)
        self.data_client = DataClient(interval=interval, warm_time=warm_time, during_time=during_time)

    @staticmethod
    def data_parser_format():
        _data_time = r'\[\d+-\d+-\d+\s+\d+:\d+:\d+.\d+\]'
        _log_level = r'\s+-\s+DEBUG\:\s+'
        _res = r'##\[\'[a-z0-9]+\',\s+[a-z]+,\s+\d+\.\d+\]##'
        return _data_time + _log_level + _res

    def data_read(self, content: str) -> list:
        return re.findall(re.compile(self._format, re.I), content)

    @staticmethod
    def parser_content(str_content: str):
        _dt = str_content.split(']')[0].split('[')[-1]
        dt = datetime.strptime(_dt, "%Y-%m-%d %H:%M:%S,%f")
        k = eval(str_content.split('##')[1])
        return dt, *k

    def data_parser(self, content: str):
        _contents = self.data_read(content)

        if len(_contents):
            self.data_client.reset_current_result()
        for _c in _contents:
            self.data_client.add_data(DataParams(*self.parser_content(_c)))

    def print_intermediate_status(self, content: str):
        self.data_parser(content)
        self.data_client.print_intermediate_state()

    def print_final_status(self, content: str):
        self.data_parser(content)
        self.data_client.print_final_state(real_time=self.real_time)
        self.data_client.print_warm_state(during_time=self.real_time - self.warm_time * 2)

    def start_stream_read(self, start_time: datetime = None):
        log.info(f"[ParserResult] Starting sync report, interval:{self.interval}s, " +
                 "intermediate state results are available for reference")
        self.data_client.update_start_time(start_time)
        self.read_client.tick_read_incremental_file(callable_object=self.print_intermediate_status)

    def finish_stream_read(self, real_time=None):
        self.real_time = real_time or self.real_time
        self.read_client.final_read_incremental_file(callable_object=self.print_final_status)
        log.info("[ParserResult] Completed sync report")
