
import click

from click.testing import CliRunner
from detection.io import ClickRemotePath


def test_handles_remote_path():
    @click.command()
    @click.option('--datapath', type=ClickRemotePath(exists=False),
                  required=True)
    def entrypoint(datapath):
        pass

    runner = CliRunner()
    result = runner.invoke(entrypoint, ['--datapath', "nothing"])
    assert result.exit_code == 0
