#!/usr/bin/env python3
"""
a bunch of helper functions for read and write data
"""
import os
import json
import numpy as np
import pandas as pd
import requests
import time

from io import BytesIO
from typing import List, Union
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None


######## JSON related ########
def read_jsonl(json_file: str) -> List:
    """
    Read json data into a list of dict.
    Each file is composed of a single object type, one JSON-object per-line.
    Args:
        json_file (str): path of specific json file
    Returns:
        data (list): list of dicts
    """
    start = time.time()
    data = []
    with open(json_file) as fin:
        for line in fin:
            line_contents = json.loads(line)
            data.append(line_contents)
    end = time.time()
    elapse = end - start
    if elapse > 1:
        print("\tLoading {} takes {:.2f} seconds.".format(
            json_file, elapse))
    return data


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            # return super(MyEncoder, self).default(obj)

            raise TypeError(
                "Unserializable object {} of type {}".format(obj, type(obj))
            )


def write_json(data: Union[list, dict], outfile: str) -> None:
    json_dir, _ = os.path.split(outfile)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir)

    with open(outfile, 'w') as f:
        json.dump(data, f, cls=JSONEncoder, ensure_ascii=False, indent=2)


def read_json(filename: str) -> Union[list, dict]:
    """read json files"""
    with open(filename, "rb") as fin:
        data = json.load(fin, encoding="utf-8")
    return data


######## image related ########
def pil_loader(path: str) -> Image.Image:
    """load an image from path, and suppress warning"""
    # to avoid crashing for truncated (corrupted images)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img.convert('RGB')


######## pd.DataFrame related ########
def save_or_append_df(out_path, df):
    if os.path.exists(out_path):
        previous_df = pd.read_csv(out_path)
        df = pd.concat([previous_df, df], ignore_index=True)
    df.to_csv(out_path)
    print(f"Saved output at {out_path}")


def read_df(path):
    data = pd.read_csv(path)

    for c in data.columns:
        if c.startswith("Unnamed"):
            del data[c]

    data.dropna(subset=['fixed_type'], inplace=True)
    data = data.replace(np.nan, "", regex=True)
    return data


############# npy related #############
def read_npy_dict(path: str) -> dict:
    # when saving a dict into .npy format,
    # need some special attention to load that object
    return np.load(path, allow_pickle=True)[()]


