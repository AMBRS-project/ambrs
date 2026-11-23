"""Quoted model invocation preserves spaces and literal shell metacharacters."""
import shlex

from ambrs import mam4, partmc
from ambrs.aerosol import AerosolProcesses


def test_executable_with_quote_and_spaces_is_a_single_argument():
    executable = "/opt/model's binaries/box model"
    prefix = "test's sample"
    mam = mam4.AerosolModel(AerosolProcesses()).invocation(executable, prefix)
    pmc = partmc.AerosolModel(AerosolProcesses(), n_part=10).invocation(executable, prefix)
    assert shlex.split(mam) == [executable]
    assert shlex.split(pmc) == [executable, prefix + ".spec"]
