import os
import os.path as osp

import hashlib

from typing import List


def listdir(root, ext) -> List[str]:
    paths = []
    for filename in os.listdir(root):
        if osp.isdir(osp.join(root, filename)):
            nested = listdir(osp.join(root, filename), ext)
            paths.extend(nested)
        elif filename.endswith(ext):
            paths.append(osp.join(root, filename))

    return paths


def check_md5sum(path, md5):
    with open(path, "rb") as f:
        data = f.read()
        file_md5 = hashlib.md5(data).hexdigest()

    return md5 == file_md5
