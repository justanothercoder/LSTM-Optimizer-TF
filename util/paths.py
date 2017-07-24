import pathlib
import subprocess
import shlex


def model_path(name):
    path = pathlib.Path('models') / name
    return path


def experiment_path(name, experiment_name, phase):
    return model_path(name) / phase / experiment_name


def snapshots_path(experiment_path):
    return experiment_path / 'snapshots'


def config_path(model_path):
    return model_path / 'config.json'


def make_dirs(*dirs):
    command = 'mkdir -p ' + ' '.join(str(s) for s in dirs)
    subprocess.call(shlex.split(command))
