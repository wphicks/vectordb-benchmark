import os
import h5py
from yaml import full_load
import json
import numpy as np
import random
from itertools import product
from sklearn import preprocessing
from utils.util_log import log


def check_file_exist(file_dir, out_put=True):
    if not os.path.isfile(file_dir):
        if out_put:
            log.info("[check_file_exist] File not exist:{}".format(file_dir))
        return False
    return True


def read_hdf5_file(file_path: str):
    try:
        if check_file_exist(file_path):
            return h5py.File(file_path)
    except Exception as e:
        raise Exception("[read_hdf5_file] Can not read hdf5 file({0}), error: {1}".format(file_path, e))


def read_ann_hdf5_file(file_path: str):
    file_list = read_hdf5_file(file_path)
    if sorted(["neighbors", "test", "train", "distances"]) != sorted(list(file_list)):
        raise Exception("[read_ann_hdf5_file] File does not contain all fields:{0}".format(list(file_list)))
    return file_list


def read_search_hdf5_file(file_path: str):
    file_list = read_hdf5_file(file_path)
    if sorted(["neighbors", "test", "train", "distances"]) != sorted(list(file_list)):
        raise Exception("[read_ann_hdf5_file] File does not contain all fields:{0}".format(list(file_list)))
    return file_list["test"]


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
    support_file_types = {"yaml": read_yaml_file, "json": read_json_file, "txt": read_txt_file, "npy": read_npy_file,
                          "hdf5": read_search_hdf5_file}
    file_type = file_path.split('.')[-1]
    if check_file_exist(file_path) and file_type in support_file_types:
        return support_file_types[file_type](file_path)
    raise Exception("[read_file] Can not read file: {}, please check.".format(file_path))


def read_search_file(file_path: str, root_path: str = ""):
    if not check_file_exist(file_path, out_put=False):
        file_path = root_path + file_path
    return read_file(file_path)


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


def normalize_data(similarity_metric_type, vectors):
    if similarity_metric_type == "IP":
        vectors = preprocessing.normalize(vectors, axis=1, norm='l2')
        vectors = vectors.astype(np.float32)
    elif similarity_metric_type == "L2":
        vectors = vectors.astype(np.float32)
    return vectors


def gen_combinations(args):
    if isinstance(args, list):
        flat = [el if isinstance(el, list) else [el] for el in args]
        return [list(x) for x in product(*flat)]
    elif isinstance(args, dict):
        flat = []
        for k, v in args.items():
            if isinstance(v, list):
                flat.append([(k, el) for el in v])
            else:
                flat.append([(k, v)])
        return [dict(x) for x in product(*flat)]
    else:
        raise TypeError("[gen_combinations] No args handling exists for %s" % type(args).__name__)


def create_folder(file_path):
    folder_path, file_name = os.path.split(file_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
