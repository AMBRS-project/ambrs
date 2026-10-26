"""Runner preserves quoted executable paths and arguments."""
import shlex
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ambrs import mam4, partmc
from ambrs.aerosol import AerosolProcesses
from ambrs.runners import PoolRunner


def test_model_invocations_escape_executable_with_spaces():
    exe = "/opt/aerosol models/box model"
    partmc_cmd = partmc.AerosolModel(AerosolProcesses(), n_part=10).invocation(exe, "case")
    mam4_cmd = mam4.AerosolModel(AerosolProcesses()).invocation(exe, "case")
    assert shlex.split(partmc_cmd) == [exe, "case.spec"]
    assert shlex.split(mam4_cmd) == [exe]


def test_runner_parses_quoted_arguments():
    exe = "/opt/aerosol models/box model"
    model = SimpleNamespace(name="fake", write_input_files=Mock(),
        invocation=Mock(return_value=f"{shlex.quote(exe)} --flag 'two words'"))
    with TemporaryDirectory() as tmp, patch(
        "ambrs.runners.subprocess.run", return_value=subprocess.CompletedProcess([], 0)
    ) as execute:
        PoolRunner(model=model, executable=exe, root=tmp, num_processes=1).run([object()])
    assert execute.call_args.args[0] == [exe, "--flag", "two words"]
