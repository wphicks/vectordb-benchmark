# vectordb-benchmark

## overview
This is an open-source benchmark for evaluating the performance of vector databases, the main functions are as follows:
1. Specify the dataset and parameters to calculate the **Search Recall**
2. Specify the search vectors and parameters, and calculate the **QPS**

## run benchmark client

* Logs of the benchmarks are stored in the ./results/result.*

* Datasets of the benchmarks are stored in the ./datasets/dataset_files/

* Configs of the benchmarks are stored in the ./configurations/*.yaml


### install dependencies:
python3 (>=3.8)

`pip install -r requirements.txt`

### run recall benchmark
> This method mainly provides the calculation of the search recall value of the server for the supported datasets and configuration parameters, 
> so as to select index parameters and search parameters with a higher recall rate.
> 
> For parameter definitions, refer to the configuration file: **./configurations/milvus_recall.yaml**

run help: `python3 main.py recall --help`

```text
Usage: main.py recall [OPTIONS]

  :param host: server host

  :param engine: only support milvus

  :param dataset_name: four datasets are available to choose from as follows:
  glove-25-angular / glove-100-angular / gist-960-euclidean / deep-image-96-angular

  :param prepare: search an existing collection without skipping data
  preparation

  :param config_name:     specify the name of the configuration file in the
  configurations directory,     and only use this configuration file;     if
  not specified, all milvus_recall*.yaml in the configuration directory will
  be used.

Options:
  --host TEXT               [default: localhost]
  --engine TEXT             [default: milvus]
  --dataset-name TEXT       [default: glove-25-angular]
  --prepare / --no-prepare  [default: prepare]
  --config-name TEXT
  --help                    Show this message and exit.
```

example: `python3 main.py recall --host localhost --engine milvus --dataset-name glove-25-angular`



### run concurrency benchmark
> This method is used to perform concurrent search operations on an existing collection and given concurrency parameters, 
> and print concurrency test results such as RPS.
> 
> For parameter definitions, refer to the configuration file: **./configurations/milvus_concurrency.yaml**

run help: `python3 main.py concurrency --help`

```text
Usage: main.py concurrency [OPTIONS]

  :param host: server host

  :param engine: only support milvus

  :param config_name:     specify the name of the configuration file in the
  configurations directory,     and only use this configuration file;     if
  not specified, all milvus_concurrency*.yaml in the configuration directory
  will be used.

Options:
  --host TEXT         [default: localhost]
  --engine TEXT       [default: milvus]
  --config-name TEXT
  --help              Show this message and exit.
```

example: `python3 main.py concurrency --host localhost --engine milvus`

* reqs: the total number of api requests
* fails: the total number of api failed requests
* Avg: average response time of interface within statistical interval
* Min: minimum response time of interface within statistical interval
* Max: maximum response time of interface within statistical interval
* Median: median response time of interface within statistical interval
* TP99: TP99 response time of interface within statistical interval
* req/s: the number of requests per second for the api in the statistical interval
* failures/s: the number of failed requests per second of the api within the statistical interval


```bash
INFO: [ParserResult] Starting sync report, interval:20s, intermediate state results are available for reference (parser_result.py:50)
INFO:  Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s (data_client.py:42)
INFO: --------------------------------------------------------------------------------------------------------------------- (data_client.py:46)
INFO:  Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s (data_client.py:42)
INFO:  search                            1467     0(0.00%)  |     131      43    1291      92     823  |   73.35    0.00 (data_client.py:44)
INFO:  Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s (data_client.py:42)
INFO:  search                            2706     0(0.00%)  |     154      47    1040     118     703  |   61.95    0.00 (data_client.py:44)
INFO:  Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s (data_client.py:42)
INFO:  search                            4209     0(0.00%)  |     137      44    1703      97    1167  |   75.15    0.00 (data_client.py:44)
INFO: [MultiProcessConcurrent] End concurrent pool (multi_process.py:49)
INFO: ------------------------------------------------- Print final status ------------------------------------------------ (data_client.py:49)
INFO:  Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s (data_client.py:50)
INFO:  search                            4279     0(0.00%)  |     139      43    1703     100     785  |   70.53    0.00 (data_client.py:52)
```

*If you want to perform a concurrency test based on the search parameter with the most appropriate recall value, you can update the search parameters of the recall scene to milvus_concurrency.yaml, and then conduct a concurrency test*
