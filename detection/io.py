import os
import click
import gcsfs

import typing as t
from pathlib import Path


def ls(path, gs_prefix="gs://"):
    if not path.startswith(gs_prefix):
        return os.listdir(path)

    fs = gcsfs.GCSFileSystem()
    return [f"{gs_prefix}{f}" for f in fs.ls(path)]


def parent(path, gs_prefix="gs://"):
    if not path.startswith(gs_prefix):
        return Path(path).parent

    posixpath = Path(path[len(gs_prefix):]).parent
    return os.path.join(gs_prefix, posixpath)


class ClickAnyPath(click.Path):
    _gs_prefix = "gs://"

    def convert(
        self,
        value: t.Any,
        param: t.Optional["click.Parameter"],
        ctx: t.Optional["click.Context"],
    ) -> t.Any:
        if not value.startswith(self._gs_prefix):
            return super().convert(value, param, ctx)

        fs = gcsfs.GCSFileSystem()
        exists = fs.exists(value)

        if exists:
            return value

        # Proceed even if not exists
        if not exists and not self.exists:
            return value

        try:
            filenames = ls(parent(value, self._gs_prefix))
            content = "\n".join(filenames)
            parent_content = f"The parent folder contains:{content}"
        except FileNotFoundError:
            parent_content = "Parent folder does not exit either"

        self.fail(
            f"The entity \n{value}\n does not exit. {parent_content}",
            param,
            ctx
        )
        return value
