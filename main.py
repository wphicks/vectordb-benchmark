import typer
from configurations import get_files
from client import ClientEntry
from common.common_func import read_file
from utils.util_log import test_log, log

app = typer.Typer()


@app.command()
def concurrency(host: str = "localhost", engine: str = typer.Option("milvus"), config: str = typer.Option("")):
    """
    :param host: server host

    :param engine: only support milvus

    :param config: specify the absolute path of the configuration file, and only use this configuration file;
    if not specified, all milvus_concurrency*.yaml in the configuration directory will be used.
    """
    configs = [config] if config != "" else get_files(f"{engine}_concurrency")
    test_log.clear_log_file()
    log.info(" Concurrency task started! ".center(120, "-"))
    for f in configs:
        c = ClientEntry(engine, host, read_file(f))
        c.start_concurrency()
        test_log.restore_debug_log_file()
    log.info(" Concurrency task finished! ".center(120, "-"))


@app.command()
def recall(host: str = typer.Option("localhost"), engine: str = typer.Option("milvus"),
           dataset_name: str = typer.Option("glove-25-angular"), prepare: bool = typer.Option(True),
           config: str = typer.Option("")):
    """
    :param host: server host

    :param engine: only support milvus

    :param dataset_name: four datasets are available to choose from as follows:
                        glove-25-angular / glove-100-angular / gist-960-euclidean / deep-image-96-angular

    :param prepare: search an existing collection without skipping data preparation

    :param config: specify the absolute path of the configuration file, and only use this configuration file;
     if not specified, all milvus_recall*.yaml in the configuration directory will be used.
    """
    configs = [config] if config != "" else get_files(f"{engine}_recall")
    test_log.clear_log_file()
    log.info(" Recall task started! ".center(120, "-"))
    for f in configs:
        c = ClientEntry(engine, host, read_file(f))
        c.start_recall(dataset_name=dataset_name, prepare=prepare)
        test_log.restore_debug_log_file()
    log.info(" Recall task finished! ".center(120, "-"))


if __name__ == "__main__":
    """
    Example: python3 main.py concurrency --host localhost --engine milvus
    """
    app()
