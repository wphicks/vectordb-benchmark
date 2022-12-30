import os
import glob
CONFIG_DIR = os.path.dirname(__file__)


def get_files(prefix: str = "", file_type: str = "yaml"):
    return glob.glob(os.path.join(CONFIG_DIR, prefix + "*." + file_type))


def get_custom_files(expr: str = ""):
    return glob.glob(os.path.join(CONFIG_DIR, expr + "*"))
