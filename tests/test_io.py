import click
from click.testing import CliRunner
from detection.io import ClickRemotePath


@click.command()
@click.option('--datapath', type=ClickRemotePath(exists=True), required=True)
def entrypoint(datapath):
    pass


def test_handles_remote_path():
    runner = CliRunner()
    result = runner.invoke(entrypoint, ['--test', str("Nothing")])
    print(result)
