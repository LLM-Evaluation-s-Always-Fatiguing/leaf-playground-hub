import json
import os
from typing import Dict, List


_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys")


def load_textual_key() -> List[Dict[str, str]]:
    textual_key_file = os.path.join(_root, "text.jsonl")
    keys = []
    with open(textual_key_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            keys.append(json.loads(line))
    return keys


def load_image_key() -> List[Dict[str, str]]:
    keys = []
    for dir_ in os.listdir(os.path.join(_root, "image")):
        dir_ = os.path.join(_root, "image", dir_)
        if os.path.isdir(dir_):
            keys.append({"Civilian": os.path.join(dir_, "Civilian.jpg"), "Spy": os.path.join(dir_, "Spy.jpg")})
    return keys


__all__ = [
    "load_textual_key",
    "load_image_key"
]
