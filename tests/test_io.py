import pytest

import click

from click.testing import CliRunner
from unittest.mock import patch
from detection.io import ClickAnyPath


@pytest.mark.parametrize("file, exists, ls, exit_code", [
    ("gs://simple/path.txt", True, ["simple/path.txt"], 0,),  # Simple GCS
    ("gs://simple/path.txx", False, ["simple/path.txt"], 2,),
    ("gs://simple/path.txt", False, [], 2,),  # Error arent exist
])
@pytest.mark.parametrize("option", [True, False])
@pytest.mark.parametrize("does_exist", [True, False])
@patch("detection.io.gcsfs.GCSFileSystem.exists")
@patch("detection.io.gcsfs.GCSFileSystem.ls", autospec=True)
def test_handles_remote_path(patch_ls, patch_exists, does_exist,
                             file, exists, ls, exit_code, option):
    patch_ls.return_value = ls
    patch_exists.return_value = does_exist or exists

    @click.command()
    @click.option('--datapath', type=ClickAnyPath(exists=option),
                  required=True)
    def entrypoint(datapath):
        pass

    runner = CliRunner()
    result = runner.invoke(entrypoint, ['--datapath', file])
    # print("The output", result.output)
    assert result.exit_code == 0 if option + exists > 1 else 2
