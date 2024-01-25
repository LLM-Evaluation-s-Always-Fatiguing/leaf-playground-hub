import json
import os
import random
from enum import Enum
from typing import Any, Dict, List, Union

from datasets import load_dataset
from pydantic import Field

from leaf_playground._config import _Config

DS_PATH = "cais/mmlu"
DS_SPLITS = Enum(
    'DatasetSplit',
    {"dev": "dev", "test": "test", "validation": "validation"}
)
DS_NAMES = Enum(
    'DatasetName',
    {
        n: n for n in json.load(
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds_names.json"), "r", encoding="utf-8")
    )
    }
)
QUESTION_COL = "question"
ANSWER_COL = "answer"


class DatasetConfig(_Config):
    dataset_name: DS_NAMES = Field(json_schema_extra={"default": getattr(DS_NAMES, "abstract_algebra")})
    dataset_split: DS_SPLITS = Field(json_schema_extra={"default": getattr(DS_SPLITS, "test")})
    num_samples: int = Field(json_schema_extra={"default": -1})

    def model_post_init(self, __context: Any) -> None:
        if self.num_samples < -1 or self.num_samples == 0:
            raise ValueError(f"num_samples should be -1 or positive, got {self.num_samples}")


def prepare_samples(ds_config: DatasetConfig) -> List[Dict[str, str]]:
    def preprocess(samples):
        sys_msg = f"The following are multiple choice questions (with answers) about {ds_config.dataset_name.value}."
        questions = samples["question"]

        choices = [
            "\n".join([f"{chr(65 + i)}: {c}" for i, c in enumerate(choice)])
            for choice in samples["choices"]
        ]

        new_questions = [
            f"{sys_msg}\n\n{question}\n{choice}" for question, choice in zip(questions, choices)
        ]

        return {"question": new_questions, "answer": samples["answer"]}

    dataset = load_dataset(
        path=DS_PATH,
        split=ds_config.dataset_split.value,
        name=ds_config.dataset_name.value,
        keep_in_memory=True
    )
    dataset = dataset.map(
        function=preprocess,
        keep_in_memory=True,
        remove_columns=[col for col in dataset.column_names if col not in [QUESTION_COL, ANSWER_COL]],
        batched=True
    )
    samples: List[Dict[str, Union[str, int]]] = dataset.to_list()
    if ds_config.num_samples != -1:
        sample_indices = list(range(len(samples)))
        samples = [dataset[i] for i in random.sample(sample_indices, min(len(sample_indices), ds_config.num_samples))]
    samples = [{"question": s["question"], "answer": chr(65 + s["answer"])} for s in samples]
    return samples


__all__ = ["QUESTION_COL", "ANSWER_COL", "DatasetConfig", "prepare_samples"]
