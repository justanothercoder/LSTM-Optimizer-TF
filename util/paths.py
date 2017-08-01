import pathlib
import subprocess
import shlex


def model_path(name):
    path = pathlib.Path('models') / name
    return path


def experiment_path(experiment_name):
    return pathlib.Path('experiments') / experiment_name


def make_dirs(*dirs):
    command = 'mkdir -p ' + ' '.join(str(s) for s in dirs)
    subprocess.call(shlex.split(command))
