import logging
import sys
import os
import shutil
from results import LOG_ROOT_DIR, LOG_FILE_DEBUG, LOG_FILE_INFO


class TestLogConfig:
    def __init__(self, log_debug=LOG_FILE_DEBUG, log_info=LOG_FILE_INFO, use_stream=True):
        self.log_debug = log_debug
        self.log_info = log_info
        self.log_level = logging.INFO
        self.handlers = []

        self.log = logging.getLogger()
        self.log.setLevel(logging.DEBUG)
        self.set_log_level(use_stream=use_stream)

    def set_log_level(self, use_stream=True):
        try:
            _format = "[%(asctime)s] - %(levelname)5s: %(message)s (%(filename)s:%(lineno)s)"
            formatter = logging.Formatter(_format)

            dh = logging.FileHandler(self.log_debug)
            dh.setLevel(logging.DEBUG)
            dh.setFormatter(formatter)
            self.log.addHandler(dh)
            self.handlers.append(dh)

            fh = logging.FileHandler(self.log_info)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.log.addHandler(fh)
            self.handlers.append(fh)

            if use_stream:
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(self.log_level)
                sh.setFormatter(formatter)
                self.log.addHandler(sh)
                self.handlers.append(sh)

        except Exception as e:
            print("Can not use %s or %s to log : %s" % (self.log_debug, self.log_info,  str(e)))

    def clear_log_file(self, log_files: list = None):
        if not log_files:
            log_files = [self.log_debug]
        if not os.path.isdir(LOG_ROOT_DIR):
            print("[clear_log_file] folder(%s) is not exist." % LOG_ROOT_DIR)
            os.makedirs(LOG_ROOT_DIR)

        for file_path in log_files:
            if os.path.isfile(file_path):
                # print("[clear_log_file] start modifying file(%s)." % file_path)
                with open(file_path, "r+") as f:
                    f.seek(0)
                    f.truncate()
                    f.write('')
                    f.close()
                # print("[clear_log_file] file(%s) modification is complete." % file_path)

    def clear_debug_log_file(self, retry_counts=9999):
        file_path = self.log_debug
        for i in range(retry_counts):
            file_path += f"_{i}"
            if not os.path.isfile(str(file_path)):
                shutil.copyfile(self.log_debug, file_path)
                self.clear_log_file()
                return True
            file_path = self.log_debug

        raise Exception("[rename_logfiles] Generated log path exists, please check:{0}".format(file_path))


test_log = TestLogConfig()
log = test_log.log
