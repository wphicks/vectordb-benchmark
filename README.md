# vectordb-benchmark

## overview
This is an open-source benchmark for evaluating the performance of vector databases, the main functions are as follows:
1. Specify the dataset and parameters to calculate the **Search Recall**
2. Specify the search vectors and parameters, and calculate the **RPS**

## run benchmark client

* Logs of the benchmarks are stored in the ./results/result.*

* Datasets of the benchmarks are stored in the ./datasets/dataset_files/

* Configs of the benchmarks are stored in the ./configurations/*.yaml


### install dependencies:
python3 (>=3.8)

`pip install -r requirements.txt`

### run recall benchmark
> This method mainly provides the calculation of the server's search recall value for the supported datasets and configuration parameters, 
> thus selecting index parameters and search parameters with a higher recall rate.
> 
> For parameter definitions, refer to the configuration file: **./configurations/\<engine\>_recall.yaml**

run help: `python3 main.py recall --help`

```text
Usage: main.py recall [OPTIONS]

  :param host: server host

  :param engine: only supports milvus / elasticsearch

  :param dataset_name: four datasets are available to choose from as follows:
  glove-25-angular / glove-100-angular / gist-960-euclidean / deep-image-96-angular /
  sift-128-euclidean

  :param prepare: search an existing collection without skipping data
  preparation

  :param config_name:     specify the name of the configuration file in the
  configurations directory by prefix matching;     if not specified, all
  milvus_recall*.yaml in the configuration directory will be used.

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
> For parameter definitions, refer to the configuration file: **./configurations/\<engine\>_concurrency.yaml**

run help: `python3 main.py concurrency --help`

```text
Usage: main.py concurrency [OPTIONS]

  :param host: server host

  :param engine: only supports milvus / elasticsearch

  :param config_name:     specify the name of the configuration file in the
  configurations directory by prefix matching;     if not specified, all
  milvus_concurrency*.yaml in the configuration directory will be used.

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
[ParserResult] Starting sync report, interval:20s, intermediate state results are available for reference
 Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s
---------------------------------------------------------------------------------------------------------------------
 Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s
 search                            4339     0(0.00%)  |      41      29     441      38      72  |  216.95    0.00
 Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s
 search                            9034     0(0.00%)  |      42      29     307      39      74  |  234.75    0.00
 Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s
 search                           13587     0(0.00%)  |      43      29     433      39     199  |  227.65    0.00
[MultiProcessConcurrent] End concurrent pool
------------------------------------------------- Print final status ------------------------------------------------
 Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s
 search                           13966     0(0.00%)  |      42      29     441      38     170  |  225.73    0.00
------------------------ Print the status without start and end warmup time:0s as a reference -----------------------
 Name                            # reqs      # fails  |     Avg     Min     Max  Median    TP99  |   req/s failures/s
 search                           13966     0(0.00%)  |      42      29     441      38     170  |  225.73    0.00
[ParserResult] Completed sync report
```

*If you want to perform a concurrency test based on the search parameter with the most appropriate recall value, you can update the search parameters of the recall scene to \<engine\>_concurrency.yaml, and then conduct a concurrency test*
