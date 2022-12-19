from dataclasses import dataclass
import numpy as np
import datetime
import os
try:
    STATS_NAME_WIDTH = max(min(os.get_terminal_size()[0] - 80, 30), 0)
except OSError:  # not a real terminal
    STATS_NAME_WIDTH = 30

from utils.util_log import log
DATA_FORMAT = (" %-" + str(STATS_NAME_WIDTH) + "s %7d %12s  | %7d %7d %7d %7d %7d  | %7.2f %7.2f")
TITLE_FORMAT = (" %-" + str(STATS_NAME_WIDTH) + "s %7s %12s  | %7s %7s %7s %7s %7s  | %7s %7s") % (
    "Name", "# reqs", "# fails", "Avg", "Min", "Max", "Median", "TP99", "req/s", "failures/s")


@dataclass
class DataParams:
    data_time: datetime
    api_name: str
    api_status: bool
    api_rt: float


class DataClient:
    def __init__(self, interval: int = 20):
        self.interval = interval
        self.current_time = datetime.datetime.utcnow()
        self.api_names = []

    def add_data(self, dp: DataParams):
        if not hasattr(self, dp.api_name):
            setattr(self, dp.api_name, DataEntry(self.interval, dp.api_name, self.current_time))
            self.api_names.append(getattr(self, dp.api_name))
        api = getattr(self, dp.api_name)
        api.add_res(dp)

    def reset_current_result(self):
        for api in self.api_names:
            api.reset_current_result()

    def print_intermediate_state(self):
        log.info(TITLE_FORMAT)
        for api in self.api_names:
            log.info(api.intermediate_state_to_string())
        if len(self.api_names) == 0:
            log.info('-' * len(TITLE_FORMAT))

    def print_final_state(self, during_time=None):
        log.info(" Print final status ".center(len(TITLE_FORMAT), '-'))
        log.info(TITLE_FORMAT)
        for api in self.api_names:
            log.info(api.final_state_to_string(during_time))


class DataEntry:
    def __init__(self, interval: int, api_name: str, current_time: datetime):
        self.interval = interval
        self.current_time = current_time
        self.api_name = api_name
        self.num_requests = 0
        self.num_success = 0
        self.num_failures = 0
        self.api_rt_list = []
        self.api_status_list = []

        self.current_requests = 0
        self.current_success = 0
        self.current_failures = 0
        self.current_rt_list = []

    def add_res(self, dp: DataParams):
        self.api_rt_list.append(dp.api_rt)
        self.current_rt_list.append(dp.api_rt)

        self.api_status_list.append(dp.api_status)

        self.num_requests += 1
        self.current_requests += 1
        if dp.api_status:
            self.num_success += 1
            self.current_success += 1
        else:
            self.num_failures += 1
            self.current_failures += 1

    @property
    def fail_ratio(self):
        try:
            return float(self.num_failures) / self.num_requests
        except ZeroDivisionError:
            if self.num_failures > 0:
                return 1.0
            else:
                return 0.0

    def reset_current_result(self):
        self.current_requests = 0
        self.current_success = 0
        self.current_failures = 0
        self.current_rt_list = []

    def intermediate_state_to_string(self):
        return DATA_FORMAT % (
            self.api_name,
            self.num_requests,
            "%d(%.2f%%)" % (self.num_failures, self.fail_ratio * 100),
            np.mean(self.current_rt_list),
            np.min(self.current_rt_list) or 0,
            np.max(self.current_rt_list),
            np.median(self.current_rt_list) or 0,
            np.percentile(self.current_rt_list, 99) or 0,
            self.current_requests / self.interval or 0,
            self.current_failures / self.interval or 0,
        )

    def final_state_to_string(self, during_time=None):
        delta_time = during_time or (datetime.datetime.utcnow() - self.current_time).total_seconds()
        return DATA_FORMAT % (
            self.api_name,
            self.num_requests,
            "%d(%.2f%%)" % (self.num_failures, self.fail_ratio * 100),
            np.mean(self.api_rt_list),
            np.min(self.api_rt_list) or 0,
            np.max(self.api_rt_list),
            np.median(self.api_rt_list) or 0,
            np.percentile(self.api_rt_list, 99) or 0,
            self.num_requests / delta_time or 0,
            self.num_failures / delta_time or 0,
        )
