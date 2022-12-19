import os
from yaml import full_load
import json
import numpy as np
import random
from utils.util_log import log


def check_file_exist(file_dir):
    if not os.path.isfile(file_dir):
        log.info("[check_file_exist] File not exist:{}".format(file_dir))
        return False
    return True


def read_yaml_file(file_path: str):
    try:
        with open(file_path) as f:
            file_dict = full_load(f)
    except Exception as e:
        raise Exception("[read_yaml_file] Can not open yaml file({0}), error: {1}".format(file_path, e))
    finally:
        f.close()
    return file_dict


def read_json_file(file_path: str):
    try:
        with open(file_path) as f:
            file_dict = json.load(f)
    except Exception as e:
        raise Exception("[read_json_file] Can not open json file({0}), error: {1}".format(file_path, e))
    finally:
        f.close()
    return file_dict


def read_txt_file(file_path: str):
    try:
        with open(file_path, "r+") as f:
            file_dict = f.read()
    except Exception as e:
        raise Exception("[read_txt_file] Can not open txt file({0}), error: {1}".format(file_path, e))
    finally:
        f.close()
    return eval(file_dict)


def read_npy_file(file_path: str):
    try:
        return np.load(file_path).tolist()
    except Exception as e:
        raise Exception("[read_npy_file] Can not open npy file({0}), error: {1}".format(file_path, e))


def read_file(file_path: str):
    support_file_types = {"yaml": read_yaml_file, "json": read_json_file, "txt": read_txt_file, "npy": read_npy_file}
    file_type = file_path.split('.')[-1]
    if check_file_exist(file_path) and file_type in support_file_types:
        return support_file_types[file_type](file_path)
    raise Exception("[read_file] Can not read file: {}, please check.".format(file_path))


def milvus_gen_vectors(nb, dim):
    return [[random.random() for _ in range(int(dim))] for _ in range(int(nb))]


def modify_file(file_path, input_content: str = '', is_modify=False):
    folder_path, file_name = os.path.split(file_path)
    if not os.path.isdir(folder_path):
        log.debug("[modify_file] folder(%s) is not exist." % folder_path)
        os.makedirs(folder_path)

    if not os.path.isfile(file_path):
        open(file_path, "a").close()
        log.debug("[modify_file] file(%s) is not exist." % file_path)
    else:
        if is_modify is True:
            log.debug("[modify_file] start modifying file(%s)..." % file_path)
            with open(file_path, "r+") as f:
                f.seek(0)
                f.truncate()
                f.write(input_content)
                f.close()
        else:
            with open(file_path, "a+") as f:
                f.write(input_content + '\n')
                f.close()


def write_existed_file(file_path: str, input_content: str):
    with open(file_path, "a+") as f:
        f.write(input_content + '\n')
        f.close()
