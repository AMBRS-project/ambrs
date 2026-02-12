"""ambrs.runners -- data types and functions related to running scenarios"""

from . import analysis

import logging
import math
import multiprocessing
import multiprocessing.dummy
import os
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


class PoolRunner:
    """PoolRunner: a simple scenario runner that runs scenarios in parallel
within a given root directory using a process pool of a given size

Required parameters:
    * model: an instance of a supported AMBRS aerosol model
    * executable: an absolute path to the executable for the model
    * root: an absolute path to the directory in which the scenarios are run.
            Each scenario is run in its own subdirectory, which is named either
            after the 1-based index of the scenario or using the scenario_name
            optional parameter

Optional parameters:
    * num_processes: the number of processes in the process pool, which
                     determines how many scenarios can be run in parallel
                     (default: number of available cpus)
    * scenario_name: a formatting string used to name each scenario, containing
                     an {index} parameter which the runner replaces with the
                     1-based scenario index (default: '{index}')

Chunking support (optional):
    * scenario_start: 1-based global member index for directory numbering.
                      Default 1 preserves legacy behavior.
    * max_num_digits: fixed padding width. If not provided, computed from
                      the highest index in this run (legacy behavior).
    """

    def __init__(
        self,
        model,
        executable: str,
        root: str,
        num_processes: int = None,
        scenario_name: str = "{index}",
        scenario_start: int = 1,
        max_num_digits: int | None = None,
    ):
        self.model = model
        self.executable = executable
        self.root = root
        self.num_processes = num_processes if num_processes else multiprocessing.cpu_count()
        self.scenario_name = scenario_name

        self.scenario_start = int(scenario_start)
        if self.scenario_start < 1:
            raise ValueError("scenario_start must be >= 1")

        self.max_num_digits = int(max_num_digits) if max_num_digits is not None else None

        if not os.path.exists(self.root):
            raise OSError(f"root path '{self.root}' does not exist!")

    def run(self, inputs: list[Any]) -> list[analysis.Output]:
        """runner.run(inputs) -> runs a list of scenario inputs within the
        runner's root directory, generating a directory for each of the scenarios
        """
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list of scenario inputs")

        num_inputs = len(inputs)
        if num_inputs == 0:
            return []

        # Determine padding width.
        # Legacy behavior: pad based on number of inputs (01..N).
        # Chunking behavior: pad based on global total (001..TOTAL) via max_num_digits.
        if self.max_num_digits is not None:
            pad_width = self.max_num_digits
        else:
            # This preserves original logic for single-shot runs.
            pad_width = math.floor(math.log10(num_inputs)) + 1 if num_inputs > 0 else 1

        logger.info(f"{self.model.name}: generating input for {num_inputs} scenarios...")

        found_dir = False
        args = []

        for i, input in enumerate(inputs):
            global_index = self.scenario_start + i

            # zero-pad the 1-based scenario index
            idx_str = str(global_index)
            formatted_index = idx_str.zfill(pad_width)
            scenario_name = self.scenario_name.format(index=formatted_index)

            # make the scenario directory if needed
            dir = os.path.join(self.root, scenario_name)
            if os.path.exists(dir):
                found_dir = True
            else:
                os.mkdir(dir)

            # write input files and define commands
            self.model.write_input_files(input, dir, scenario_name)
            command = self.model.invocation(self.executable, scenario_name)

            args.append({"command": command, "dir": dir})

        if found_dir:
            logger.warning(
                f"{self.model.name}: one or more existing scenario directories found. Overwriting contents..."
            )
        logger.info(f"{self.model.name}: finished generating scenario input.")

        # now run scenarios in parallel
        pool = multiprocessing.dummy.Pool(self.num_processes)
        logger.info(
            f"{self.model.name}: running {num_inputs} inputs ({self.num_processes} parallel processes)"
        )

        error_state = {"error": False}  # mutable so callback can update it

        def callback(completed_processes) -> None:
            if not all([p.returncode == 0 for p in completed_processes]):
                error_state["error"] = True

        def run_scenario(args) -> subprocess.CompletedProcess:
            f_stdout = open(os.path.join(args["dir"], "stdout.log"), "w")
            f_stderr = open(os.path.join(args["dir"], "stderr.log"), "w")
            return subprocess.run(
                args["command"].split(),
                close_fds=True,
                cwd=args["dir"],
                stdout=f_stdout,
                stderr=f_stderr,
            )

        results = pool.map_async(run_scenario, args, callback=callback)
        results.wait()

        logger.info(f"{self.model.name}: completed runs.")
        if error_state["error"]:
            logger.error(f"{self.model.name}: At least one run failed.")

        # NOTE: outputs collection remains commented out as in your original.
        return []
