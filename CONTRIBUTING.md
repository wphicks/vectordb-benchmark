# Contributing to vectordb-benchmark
Contributions to **vectordb-benchmark** are welcome from everyone.
We strive to make the contribution process simple and straightforward.

The following are a set of guidelines for contributing to **vectordb-benchmark**.
Following these guidelines makes contributing to this project easy and transparent.
These are mostly guidelines, not rules.
Use your best judgment, and feel free to propose changes to this document in a pull request.


## GitHub workflow
Generally, we follow the "fork-and-pull" Git workflow.

1. [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) the repository on GitHub.
2. Clone your fork to your local machine with `git clone git@github.com:<yourname>/vectordb-benchmark.git.`
3. Create a branch with `git checkout -b my-topic-branch`.
4. [Commit](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/committing-changes-to-a-pull-request-branch-created-from-a-fork) changes to your own branch, then push to GitHub with `git push origin my-topic-branch`.
5. [Submit](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) a pull request so that we can review your changes.

Remember to sync your forked repository before submitting proposed changes upstream. If you have an existing local repository, please update it before you start, to minimize the chance of merge conflicts.

```
git remote add upstream git@github.com:zilliztech/vectordb-benchmark.git
git fetch upstream
git checkout upstream/master -b my-topic-branch
```

## Style guide
We generally follow the [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/) - this applies to the main zilliztech/vectordb-benchmark repo on Github as well as code uploaded to our vectordb-benchmark Hub.


## How to collaborate
### register a dataset
Add the new dataset to the `dataset_configs` dictionary in `./datasets/dataset_configs.py`, format as follows:
```
"<dataset name>": {
    "dim": <int>,
    "link": "<the address to download the dataset>",
    "similarity_metric_type": "<similarity type>"
}
```

### add a new engine
**The main code can refer to the engine client already existed**

1. Create an engine subdirectory under the `./client`,
and perform [interface encapsulation](client/base/interface.py), [input parameter processing](client/base/parameters.py), and [client encapsulation](client/base/client_base.py),
and respectively inherit the parent class in the base.
2. Add the new engine to `.client/__init__.py`
3. Add the default configuration files to `.configurations/<engine>_<concurrency or recall>.yaml`
4. Add docker-compose deployment configuration in `.server/<engine>`
