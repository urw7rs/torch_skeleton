import os
import os.path as osp

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
