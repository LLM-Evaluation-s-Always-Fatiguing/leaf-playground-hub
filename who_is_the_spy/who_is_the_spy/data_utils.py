import json
from os import listdir
from os.path import abspath, isdir, join
from typing import Dict, List

from leaf_playground_cli.utils.path_utils import get_dataset_dir


key_dir = join(get_dataset_dir(abspath(__file__)), "keys")


def load_textual_key() -> List[Dict[str, str]]:
    textual_key_file = join(key_dir, "text.jsonl")
    keys = []
    with open(textual_key_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            keys.append(json.loads(line))
    return keys


def load_image_key() -> List[Dict[str, str]]:
    image_dir = join(key_dir, "image")
    keys = []
    for dir_ in listdir(image_dir):
        dir_ = join(image_dir, dir_)
        if isdir(dir_):
            keys.append({"Civilian": join(dir_, "Civilian.jpg"), "Spy": join(dir_, "Spy.jpg")})
    return keys


__all__ = [
    "load_textual_key",
    "load_image_key"
]
