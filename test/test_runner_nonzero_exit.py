"""A nonzero exit status must not look like a successful simulation."""
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import subprocess
import pytest

from ambrs.runners import PoolRunner


def test_runner_raises_for_nonzero_process_exit():
    model = SimpleNamespace(
        name="fake", write_input_files=Mock(), invocation=Mock(return_value="fake-command")
    )
    with TemporaryDirectory() as tmp, patch(
        "ambrs.runners.subprocess.run",
        return_value=subprocess.CompletedProcess(["fake-command"], returncode=7),
    ):
        runner = PoolRunner(model=model, executable="fake-command", root=tmp, num_processes=1)
        with pytest.raises(RuntimeError, match="At least one run failed"):
            runner.run([object()])
