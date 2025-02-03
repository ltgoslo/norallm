# Copyright (c) 2023, ALIBABA CORPORATION.  All rights reserved.

import os

CHECKPOINT_PREFIX = "checkpoint-"


def print_rank(rank, *args, **kwargs):
    if os.environ.get("RANK") == str(rank):
        print(*args, **kwargs, flush=True)


def ensure_local_directory_exists(path):
    """
    Build path's containing directory if it does not already exist.
    """
    path = os.path.dirname(path)
    os.makedirs(path, exist_ok=True)


class FileSystem(object):
    def __init__(self, root_dir: str, ckpt_relative_dir: str):
        self.root_dir = root_dir
        assert ckpt_relative_dir is not None
        self.ckpt_dir = os.path.join(root_dir, ckpt_relative_dir, "")

    def get_root_dir(self):
        return self.root_dir

    def get_checkpoint_dir(self):
        return self.ckpt_dir

    def exists(self, path, is_path_under_root=False):
        pass

    def get_filelike_writer(self, path: str, mode="w", is_path_under_root=False):
        pass

    def get_filelike_reader(self, path: str, mode="r", is_path_under_root=False):
        pass

    def _get_full_path(self, path, is_path_under_root):
        if is_path_under_root:
            return os.path.join(self.root_dir, path)
        return os.path.join(self.ckpt_dir, path)


class EmptyFileSystem(FileSystem):
    def __init__(self):
        super().__init__("", "")
        print(f"WARNING: EmptyFileSystem is used")

    def exists(self, path, is_path_under_root=False):
        return False

    def get_filelike_writer(self, path: str, mode="w", is_path_under_root=False):
        assert False, "Unexpected writer call"

    def get_filelike_reader(self, path: str, mode="r", is_path_under_root=False):
        assert False, "Unexpected reader call"


class LocalFileSystem(FileSystem):
    def __init__(self, root_dir: str, ckpt_relative_dir: str):
        super().__init__(root_dir, ckpt_relative_dir)
        print(f"LocalFileSystem: {root_dir} {ckpt_relative_dir}")

    def exists(self, path, is_path_under_root=False):
        return os.path.exists(self._get_full_path(path, is_path_under_root))

    def get_filelike_writer(self, path: str, mode="w", is_path_under_root=False):
        full_path = self._get_full_path(path, is_path_under_root)
        ensure_local_directory_exists(full_path)
        return open(full_path, mode)

    def get_filelike_reader(self, path: str, mode="r", is_path_under_root=False):
        return open(self._get_full_path(path, is_path_under_root), mode)


def create_read_file_system(args, relative_dir, custom_root_dir=None):
    root_dir = custom_root_dir if custom_root_dir is not None else args.load
    assert args.load is not None
    return LocalFileSystem(root_dir=root_dir, ckpt_relative_dir=relative_dir if relative_dir is not None else "")


def create_write_file_system(args, relative_dir: str):
    assert args.save is not None
    local_dir = os.path.join(
        args.save, relative_dir, ""
    )  # '' will add trailing slash
    ensure_local_directory_exists(local_dir)
    return LocalFileSystem(root_dir=args.save, ckpt_relative_dir=relative_dir)
