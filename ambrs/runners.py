"""ambrs.runners -- data types and functions related to running scenarios"""

import multiprocessing as mp
import os

class PoolRunner:
    """PoolRunner: a simple scenario runner that runs scenarios in parallel
within a given root directory using a process pool of a given size
Required parameters:
    * executable: an absolute path to an executable aerosol model used to run
                  scenarios
    * invocation: a formatting string defining how the executable is invoked,
                  containing the following formatting parameters, which are
                  substituted with their values by the runner:
        * {exe} the path to the executable program
        * {prefix} the input file prefix
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
                     scenario index (default: '{index}')
    """
    def __init__(self,
                 executable: str,
                 invocation: str,
                 root: str,
                 num_processes: int = None,
                 scenario_name: str = '{index}'):
        self.executable = executable
        self.invocation = invocation
        self.root = root
        if num_processes:
            self.num_processes = num_processes
        else:
            self.num_processes = mp.cpu_count()
        self.scenario_name = scenario_name


    def run(self, inputs):
        """runner.run(inputs) -> runs a list of scenario inputs within the
runner's root directory, generating a directory for each of the scenarios"""
        if not isinstance(inputs, list):
            raise TypeError('inputs must be a list of scenario inputs')

        # make scenario directories and populate them with input files
        num_inputs = len(inputs)
        commands = []
        for i, input in enumerate(inputs):
            scenario_name = self.scenario_name.format(index = i)
            dir = os.path.join(self.root, scenario_name)
            os.mkdir(dir)
            input.write_files(dir, scenario_name)
            commands.append(self.invocation.format(exe = self.executable,
                                                   prefix = scenario_name))

        # now run all the things!
        mp.set_start_method('spawn')
        num_processes = min(self.num_processes, num_inputs)
        with mp.Pool(processes = num_processes) as pool:
            scenario_name = self.scenario_name.format(index = i)
            dir = os.path.join(self.root, scenario_name)
            # run them all unordered to maximize throughput, since they're just
            # writing files to the filesystem anyway
            pool.imap_unordered(os.system, commands)
