"""A successful executable exit does not trigger runner failure handling."""
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from ambrs.runners import PoolRunner


def test_zero_exit_still_returns_empty_outputs():
    model = SimpleNamespace(name="fake", write_input_files=Mock(),
                            invocation=Mock(return_value="fake-command"))
    with TemporaryDirectory() as root, patch(
        "ambrs.runners.subprocess.run",
        return_value=subprocess.CompletedProcess(["fake-command"], returncode=0),
    ):
        runner = PoolRunner(model=model, executable="fake-command", root=root, num_processes=1)
        assert runner.run([object()]) == []
