import typer
from configurations import get_files, get_custom_files
from client import ClientEntry
from common.common_func import read_file
from utils.util_log import log

app = typer.Typer()


@app.command()
def concurrency(host: str = "localhost", engine: str = typer.Option("milvus"), config_name: str = typer.Option("")):
    """
    :param host: server host

    :param engine: only supports milvus / elasticsearch

    :param config_name:
        specify the name of the configuration file in the configurations directory by prefix matching;
        if not specified, all <engine>_concurrency*.yaml in the configuration directory will be used.
    """
    configs = get_custom_files(config_name) if config_name != "" else get_files(f"{engine}_concurrency")
    log.clear_log_file()
    log.info(" Concurrency task started! ".center(120, "-"))
    for f in configs:
        c = ClientEntry(engine, host, read_file(f))
        c.start_concurrency()
        log.restore_debug_log_file()
    log.info(" Concurrency task finished! ".center(120, "-"))


@app.command()
def recall(host: str = typer.Option("localhost"), engine: str = typer.Option("milvus"),
           dataset_name: str = typer.Option("glove-25-angular"), prepare: bool = typer.Option(True),
           config_name: str = typer.Option("")):
    """
    :param host: server host

    :param engine: only supports milvus / elasticsearch

    :param dataset_name: four datasets are available to choose from as follows:
                        glove-25-angular / glove-100-angular / gist-960-euclidean / deep-image-96-angular /
                        sift-128-euclidean

    :param prepare: search an existing collection without skipping data preparation

    :param config_name:
        specify the name of the configuration file in the configurations directory by prefix matching;
        if not specified, all <engine>_recall*.yaml in the configuration directory will be used.
    """
    configs = get_custom_files(config_name) if config_name != "" else get_files(f"{engine}_recall")
    log.clear_log_file()
    log.info(" Recall task started! ".center(120, "-"))
    for f in configs:
        c = ClientEntry(engine, host, read_file(f))
        c.start_recall(dataset_name=dataset_name, prepare=prepare)
        log.restore_debug_log_file()
    log.info(" Recall task finished! ".center(120, "-"))


if __name__ == "__main__":
    """
    Example: python3 main.py concurrency --host localhost --engine milvus
    """
    app()
