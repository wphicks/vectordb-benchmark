import os
import requests
import urllib.request
from utils.util_log import log
from common.common_func import check_file_exist, create_folder
from datasets.dataset_configs import DatasetConfig


class DatasetDownload:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def download(self):
        if check_file_exist(self.config.path) and self.check_file_length():
            log.info(f"[DatasetDownload] File existed:{self.config.path}")
        else:
            create_folder(self.config.path)
            filename, headers = urllib.request.urlretrieve(url=self.config.link, filename=self.config.path)
            log.info(f"[DatasetDownload] File has been downloaded:{filename}")
            # if not self.check_file_length():
            #     raise Exception("[DatasetDownload] File check failed:{}".format(self.config.path))
        return self.config.path

    def check_file_length(self):
        if self.config.link.startswith('file://'):
            return True
        true_length = int(requests.head(self.config.link).headers.get("Content-Length", -1))
        real_length = os.stat(self.config.path).st_size

        if real_length == true_length:
            return True
        log.info(f"[DatasetDownload] File check failed, true_length:{true_length}, real_length: {real_length}")
        return False
