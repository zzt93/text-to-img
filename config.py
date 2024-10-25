import os
from enum import IntEnum


class PathType(IntEnum):
    train = 0
    test = 1
    model = 2
    result = 3


def path(path_type: PathType, root_dir: str, filename: str = "feature.pt") -> str:
    if path_type == PathType.train:
        dir = os.path.join(root_dir, "train")
        if not os.path.exists(dir):
            raise Exception("no train data dir")
        if filename:
            return os.path.join(dir, filename)
        return dir
    elif path_type == PathType.test:
        dir = os.path.join(root_dir, "test")
        if not os.path.exists(dir):
            raise Exception("no test data dir")
        return dir
    elif path_type == PathType.model:
        dir = os.path.join(root_dir, "model")
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    elif path_type == PathType.result:
        dir = os.path.join(root_dir, "result")
        if not os.path.exists(dir):
            os.makedirs(dir)
        return os.path.join(dir, filename)
