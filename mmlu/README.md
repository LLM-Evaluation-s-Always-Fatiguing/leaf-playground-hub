## Introduction

This scenario simulation project using MMLU dataset to evaluate LLM-based agents' ability on different tasks.

There is one examiner agent (a static agent) who downloads the MMLU dataset from Hugging Face Datasets Hub.

In each round, the examiner agent broadcasts one sample to all examinee agents (a kind of dynamic agent whose number is not limited), each examinee agent need to choose an answer that it thinks is correct to respond.

This project implemented a bunch of auto evaluators that can automatically evaluate whether an examinee agent's answer is correct. Also, it supports manual evaluation.

When all samples are answered by all examinees, a reporter will generate a bar chart to show each examinee's answer accuracy.

> It's highly recommend to read this project's [source code](https://github.com/LLM-Evaluation-s-Always-Fatiguing/leaf-playground-hub/tree/main/mmlu) or use it as a starter if you want to implement a project that uses a dataset to evaluate LLM-based agents.

## Dependencies

Make sure you have all the additional required packages listed below installed in your environment before using this project:

```txt
# requirements.txt
datasets
```

You can copy above dependencies to a `requirements.txt` file and run `pip install -r requirements.txt` to install those dependencies.
