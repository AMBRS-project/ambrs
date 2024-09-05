"""ambrs.runners -- data types and functions related to running scenarios"""

import logging
import multiprocessing
import multiprocessing.dummy
import os
import subprocess

logger = logging.getLogger(__name__)

class PoolRunner:
    """PoolRunner: a simple scenario runner that runs scenarios in parallel
within a given root directory using a process pool of a given size
Required parameters:
    * executable: an absolute path to an executable aerosol model used to run
                  scenarios
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
    """
    def __init__(self,
                 name: str,
                 executable: str,
                 root: str,
                 num_processes: int = None,
                 scenario_name: str = '{index}'):
        self.name = name
        self.executable = executable
        self.root = root
        self.num_processes = num_processes
        self.scenario_name = scenario_name

        if not os.path.exists(self.root):
            raise OSError(f"root path '{self.root}' does not exist!")

    def run(self, inputs):
        """runner.run(inputs) -> runs a list of scenario inputs within the
runner's root directory, generating a directory for each of the scenarios"""
        if not isinstance(inputs, list):
            raise TypeError('inputs must be a list of scenario inputs')

        # make scenario directories, populate them with input files, and set
        # up arguments to subprocesses
        num_inputs = len(inputs)
        logger.info(f'{self.name}: writing {num_inputs} sets of input files...')
        found_dir = False
        args = []
        for i, input in enumerate(inputs):
            scenario_name = self.scenario_name.format(index = i+1)
            dir = os.path.join(self.root, scenario_name)
            if os.path.exists(dir):
                found_dir = True
            else:
                os.mkdir(dir)
            input.write_files(dir, scenario_name)
            command = input.invocation(self.executable, scenario_name)
            dir = os.path.join(self.root, scenario_name)
            args.append({
                'command': command,
                'name': scenario_name,
                'dir': dir
            })
        if found_dir:
            logger.warning(f'{self.name}: one or more existing scenario directories found. Overwriting contents...')
        logger.info(f'{self.name}: completed writing input files.')

        # now run scenarios in parallel
        pool = multiprocessing.dummy.Pool(self.num_processes)
        logger.info(f'{self.name}: running {num_inputs} inputs...')

        error_occurred = False
        def callback(completed_processes) -> None:
            if not all([p.returncode == 0 for p in completed_processes]):
                error_occurred = True
        def run_scenario(args) -> subprocess.CompletedProcess:
            f_stdout = open(os.path.join(args['dir'], 'stdout.txt'), 'w')
            f_stderr = open(os.path.join(args['dir'], 'stderr.txt'), 'w')
            return subprocess.run(args['command'].split(),
                close_fds = True,
                cwd = args['dir'],
                stdout = f_stdout,
                stderr = f_stderr,
            )
        results = pool.map_async(run_scenario, args, callback = callback)
        results.wait()

        logger.info(f'{self.name}: completed runs.')
        if error_occurred:
            logger.error('f{self.name}: At least one run failed.')
