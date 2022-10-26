import os
import os.path as osp

import gdown
import wget

import hashlib

import zipfile

from typing import List, Optional


def listdir(root, ext) -> List[str]:
    paths = []
    for filename in os.listdir(root):
        if osp.isdir(osp.join(root, filename)):
            nested = listdir(osp.join(root, filename), ext)
            paths.extend(nested)
        elif filename.endswith(ext):
            paths.append(osp.join(root, filename))

    return paths


def downloaded(path):
    downloaded = osp.exists(path)

    if downloaded:
        print(f"{osp.basename(path)} exists, skipping download")

    return downloaded


def download_from_gdrive(url, path, verbose: Optional[bool] = None):
    if verbose is None:
        quiet = False
    else:
        quiet = not verbose

    makedirs(osp.dirname(path))
    output = gdown.download(url, output=path, quiet=quiet)
    if output is None:
        raise ConnectionError(f"Downloading to {path} from {url} failed")

    return output


def download_url(url, path, verbose: Optional[bool] = None):
    if verbose is None:
        verbose = True

    if verbose:
        print(f"Downloading {osp.basename(path)}")

    makedirs(osp.dirname(path))
    out = wget.download(url, out=path)
    return out


def check_md5sum(path, md5):
    with open(path, "rb") as f:
        data = f.read()
        file_md5 = hashlib.md5(data).hexdigest()

    return md5 == file_md5


def makedirs(path):
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_zip(src, dst):
    makedirs(dst)
    with zipfile.ZipFile(src, "r") as zip_ref:
        zip_ref.extractall(dst)


def listzip(path, ext) -> List[str]:
    paths = []
    with zipfile.ZipFile(path, "r") as zip_ref:
        for path_obj in zip_ref.filelist:
            path = path_obj.filename
            if path.endswith(ext):
                paths.append(path)

    return paths
