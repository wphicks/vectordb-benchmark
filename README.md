# vectordb-benchmark

## overview
This tool provides the ability to calculate the performance of the vector database, the main functions are as follows:
1. Specify the data set and parameters to calculate the search recall
2. Specify the search vector and parameters, and calculate the QPS

## run benchmark client
* Logs of the benchmarks are stored in the ./results/result.*

* Datasets of the benchmarks are stored in the ./datasets/dataset_files/

* Configs of the benchmarks are stored in the ./configurations/*.yaml


### install dependencies:

`pip install -r requirements.txt`

### run recall benchmark:

example: `python3 main.py recall --host localhost --engine milvus --dataset-name glove-25-angular`

run help: `python3 main.py recall --help`

```text
Usage: main.py recall [OPTIONS]

  :param host: server host

  :param engine: only support milvus

  :param dataset_name: four datasets are available to choose
  from as follows:                     glove-25-angular /
  glove-100-angular / gist-960-euclidean / deep-image-96-angular

  :param prepare: search an existing collection without skipping
  data preparation

  :param config: specify config file path

Options:
  --host TEXT               [default: localhost]
  --engine TEXT             [default: milvus]
  --dataset-name TEXT       [default: glove-25-angular]
  --prepare / --no-prepare  [default: prepare]
  --config TEXT
  --help                    Show this message and exit.
```


### run concurrency benchmark:

example: `python3 main.py concurrency --host localhost --engine milvus`

run help: `python3 main.py concurrency --help`

```text
Usage: main.py concurrency [OPTIONS]

  :param host: server host

  :param engine: only support milvus

  :param config: specify config file path

Options:
  --host TEXT    [default: localhost]
  --engine TEXT  [default: milvus]
  --config TEXT
  --help         Show this message and exit.
```