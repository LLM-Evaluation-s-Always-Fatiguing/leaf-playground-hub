## Introduction

This scenario simulation project support using [ragas](https://github.com/explodinggradients/ragas) to evaluate your Retrieval Augmented Generation (RAG) pipelines.

The project support downloading any NLP datasets (theoretically, and you need to carefully configure the dataset config to make sure the dataset can be properly preprocessed so that it can be used by the examiner agent) from Hugging Face Datasets Hub.

An examiner agent (a static agent) downloads specified dataset and preprocess it based on the given dataset config; in each round, the examiner agent broadcast one question to all examinee agents (who use LLM as backend and whose core workflow is a RAG pipeline); each examinee agent answer to the question and provide references it searched.

For each response provided by an examinee agent, a ragas based evaluator (if triggered) will automatically evaluate the quality of the examinee agent's answer and references it searched.

Below are metrics that ragas supports, and you can select some of them (or all of them) to evaluate each examinee's performance:
- answer_correctness: measures answer correctness compared to ground truth as a combination of factuality and semantic similarity.
- answer_relevancy: scores the relevancy of the answer according to the given question. answers with incomplete, redundant or unnecessary information is penalized. score can range from 0 to 1 with 1 being the best.
- answer_similarity: scores the semantic similarity of ground truth with generated answer. cross encoder score is used to quantify semantic similarity. for more detailed information, you can read the [SAS paper](https://arxiv.org/pdf/2108.06130.pdf)
- context_precision: average Precision is a metric that evaluates whether all of the relevant items selected by the model are ranked higher or not.
- context_recall: estimates context recall by estimating TP and FN using annotated answer and retrieved context.
- context_relevancy: extracts sentences from the context that are relevant to the question with self-consistancy checks. the number of relevant sentences and is used as the score.
- faithfulness: measures the factual consistency of the generated answer against the given context. tt is calculated from answer and retrieved context. the answer is scaled to (0,1) range. higher the better.

<!-- 
# un-comment when chart is implemented

When all samples are answered by all examinees, a reporter will generate a chart to show each examinee's performance on selected metrics.

-->

## Dependencies

Make sure you have all the additional required packages listed below installed in your environment before using this project:

```txt
# requirements.txt
datasets
ragas
```

You can copy above dependencies to a `requirements.txt` file and run `pip install -r requirements.txt` to install those dependencies.