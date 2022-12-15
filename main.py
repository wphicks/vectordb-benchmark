import typer

from configurations import get_files
from common.common_func import read_file
from client import ClientEntry
from utils.util_log import test_log, log
from concurrency.multi_process import MultiProcessConcurrent


def run(host: str = "localhost", engine: str = typer.Option("milvus")):
    """
    Example: python3 main.py --host localhost --engine milvus
    """
    test_log.clear_log_file()
    for f in get_files(engine):
        c = ClientEntry(engine, host, read_file(f))
        c.get_concurrent_start_params()
        MultiProcessConcurrent().start(c.client)
        test_log.clear_debug_log_file()


if __name__ == "__main__":
    typer.run(run)
