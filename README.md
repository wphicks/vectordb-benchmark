# vectordb-benchmark

## overview
This tool provides the ability to calculate the performance of the vector database, the main functions are as follows:
1. Specify the data set and parameters to calculate the search recall
2. Specify the search vector and parameters, and calculate the QPS

## run benchmark client
Logs of the benchmarks are stored in the ./results/result.*
Datasets of the benchmarks are stored in the ./datasets/dataset_files/

install dependencies:

`pip install -r requirements.txt`

run benchmark:

recall: `python3 main.py recall --host localhost --engine milvus --dataset-name glove-25-angular`

concurrency: `python3 main.py concurrency --host localhost --engine milvus`