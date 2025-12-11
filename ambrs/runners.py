"""ambrs.runners -- data types and functions related to running scenarios"""

from . import analysis

import logging
import math
import multiprocessing
import multiprocessing.dummy
import os
import subprocess
from typing import Any

import numpy as np
import timeit

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
    """
    def __init__(self, model,
                 executable: str,
                 root: str,
                 num_processes: int = None,
                 scenario_name: str = '{index}'):
        self.model = model
        self.executable = executable
        self.root = root
        self.num_processes = num_processes if num_processes else multiprocessing.cpu_count()
        self.scenario_name = scenario_name
        
        if not os.path.exists(self.root):
            raise OSError(f"root path '{self.root}' does not exist!")

#     def run(self, inputs: list[Any]) -> list[analysis.Output]:
#         """runner.run(inputs) -> runs a list of scenario inputs within the
# runner's root directory, generating a directory for each of the scenarios"""
#         if not isinstance(inputs, list):
#             raise TypeError('inputs must be a list of scenario inputs')

#         # prep scenarios to run
#         num_inputs = len(inputs)
#         max_num_digits = math.floor(math.log10(num_inputs)) + 1
#         logger.info(f'{self.model.name}: generating input for {num_inputs} scenarios...')
#         found_dir = False
#         args = []
#         for i, input in enumerate(inputs):

#             # zero-pad the 1-based scenario index
#             num_digits = math.floor(math.log10(i+1)) + 1
#             formatted_index = '0' * (max_num_digits - num_digits) + f'{i+1}'
#             scenario_name = self.scenario_name.format(index = formatted_index)

#             # make the scenario directory if needed
#             dir = os.path.join(self.root, scenario_name)
#             if os.path.exists(dir):
#                 found_dir = True
#             else:
#                 os.mkdir(dir)

#             # write input files and define commands
#             self.model.write_input_files(input, dir, scenario_name)
#             dir = os.path.join(self.root, scenario_name)
#             # FIXME: LMF addition; double-check
#             # command = self.model.invocation(self.executable, scenario_name)
#             cmd = self.model.invocation(self.executable, scenario_name)
#             if isinstance(cmd, tuple):
#                 command, env = cmd
#             else:
#                 command, env = cmd, None
#             args.append({"command": command, "dir": dir, "env": env})

#         if found_dir:
#             logger.warning(f'{self.model.name}: one or more existing scenario directories found. Overwriting contents...')
#         logger.info(f'{self.model.name}: finished generating scenario input.')
        
#         # now run scenarios in parallel
#         pool = multiprocessing.dummy.Pool(self.num_processes)
#         logger.info(f'{self.model.name}: running {num_inputs} inputs ({self.num_processes} parallel processes)')
        
#         error_occurred = False
        
#         # this function is called with a list of completed processes by pool.map_async
#         def callback(completed_processes) -> None:
            
#             if not all([p.returncode == 0 for p in completed_processes]):
#                 error_occurred = True

#         # this function is called with one of a mapped set of arguments by pool.map_async
#         def run_scenario(args) -> subprocess.CompletedProcess:
#             f_stdout = open(os.path.join(args['dir'], 'stdout.log'), 'w')
#             f_stderr = open(os.path.join(args['dir'], 'stderr.log'), 'w')
#             f_timer = open(os.path.join(args['dir'], 'timer.log'), 'w')
            
#             # FIXME: LMF addition; double-check
#             # env = _augment_env_for_camp(os.environ.copy())
            
#             env = None
#             if hasattr(self.model, "camp") and self.model.camp:
#                 env = self.model.camp.runtime_env()
#             args.append({"command": command, "dir": dir, "env": env})

#             start_time = timeit.default_timer()
#             subprocess_output = subprocess.run(args['command'].split(),
#                 close_fds = True,
#                 cwd = args['dir'],
#                 stdout = f_stdout,
#                 stderr = f_stderr,
#                 # FIXME: LMF addition; double-check
#                 env=(args.get('env') or os.environ), 
#                 # env=args.get('env', None)
#                 #env=env,
#             )
#             stop_time = timeit.default_timer()
#             elapsed_time = stop_time - start_time
#             f_timer.write(str(elapsed_time))
#             return subprocess_output
        
#         results = pool.map_async(run_scenario, args, callback = callback)
#         results.wait()
        
#         logger.info(f'{self.model.name}: completed runs.')
#         if error_occurred:
#             logger.error('f{self.model.name}: At least one run failed.')
        
#         # gather model output
#         # outputs = []
#         # for i, input in enumerate(inputs):
#         #     scenario_name = self.scenario_name.format(index = formatted_index)
#         #     output = self.model.read_output_files(input, args[i]['dir'], scenario_name)
#         #     outputs.append(output)
#         # return outputs

#     # FIXME: LMF revised run to correct her mistakes
#     def run(self, inputs: list[analysis.Output]) -> list[analysis.Output]:
#         if not isinstance(inputs, list):
#             raise TypeError('inputs must be a list of scenario inputs')

#         # prep scenarios to run
#         num_inputs = len(inputs)
#         max_num_digits = math.floor(math.log10(num_inputs)) + 1
#         logger.info(f'{self.model.name}: generating input for {num_inputs} scenarios...')
#         found_dir = False
#         args: list[dict[str, Any]] = []

#         for i, input in enumerate(inputs):
#             # zero-pad the 1-based index
#             num_digits = math.floor(math.log10(i+1)) + 1
#             formatted_index = '0' * (max_num_digits - num_digits) + f'{i+1}'
#             scenario_name = self.scenario_name.format(index=formatted_index)

#             # scenario dir
#             dir = os.path.join(self.root, scenario_name)
#             if os.path.exists(dir):
#                 found_dir = True
#             else:
#                 os.mkdir(dir)

#             # write inputs
#             self.model.write_input_files(input, dir, scenario_name)

#             # command + (optional) env from model
#             cmd = self.model.invocation(self.executable, scenario_name)
#             if isinstance(cmd, tuple):
#                 command, cmd_env = cmd
#             else:
#                 command, cmd_env = cmd, None

#             # merge env (for CAMP lib lookup)
#             env = os.environ.copy()
#             if hasattr(self.model, "camp") and getattr(self.model, "camp"):
#                 env.update(self.model.camp.runtime_env())

#             if cmd_env:
#                 env.update(cmd_env)

#             args.append({"command": command, "dir": dir, "env": env})

#         if found_dir:
#             logger.warning(f'{self.model.name}: one or more existing scenario directories found. Overwriting contents...')
#         logger.info(f'{self.model.name}: finished generating scenario input.')

#         # run in parallel
#         pool = multiprocessing.dummy.Pool(self.num_processes)
#         logger.info(f'{self.model.name}: running {num_inputs} inputs ({self.num_processes} parallel processes)')

#         def run_scenario(one):
#             out = subprocess.run(
#                 one["command"].split(),
#                 close_fds=True,
#                 cwd=one["dir"],
#                 stdout=open(os.path.join(one["dir"], "stdout.log"), "w"),
#                 stderr=open(os.path.join(one["dir"], "stderr.log"), "w"),
#                 env=one["env"],
#             )
#             with open(os.path.join(one["dir"], "timer.log"), "w") as f_timer:
#                 f_timer.write("0")  # (optional) wire real timing if you like
#             return out
        
#         results = pool.map(run_scenario, args)
#         logger.info(f'{self.model.name}: completed runs.')
#         if not all(p.returncode == 0 for p in results):
#             logger.error(f'{self.model.name}: at least one run failed.')



# # # FIXME: LMF addition; double-check
# # def _augment_env_for_camp(env: dict) -> dict:
# #     """Prepend the conda env's lib/ to the dynamic loader search path.
# #     - macOS: DYLD_FALLBACK_LIBRARY_PATH (preferred on modern macOS)
# #     - Linux: LD_LIBRARY_PATH
# #     """
# #     env = dict(env)  # copy
# #     conda_prefix = os.environ.get("CONDA_PREFIX", "")
# #     conda_lib = Path(conda_prefix) / "lib"
# #     if conda_lib.exists():
# #         if sys.platform == "darwin":
# #             key = "DYLD_FALLBACK_LIBRARY_PATH"
# #         elif sys.platform.startswith("linux"):
# #             key = "LD_LIBRARY_PATH"
# #         else:
# #             return env  # no-op on Windows
# #         current = env.get(key, "")
# #         env[key] = f"{conda_lib}:{current}" if current else str(conda_lib)
# #     return env

    def run(self, inputs: list[Any]) -> list[analysis.Output]:
        """runner.run(inputs) -> runs a list of scenario inputs within the
runner's root directory, generating a directory for each of the scenarios"""
        if not isinstance(inputs, list):
            raise TypeError('inputs must be a list of scenario inputs')
        
        # prep scenarios to run
        
        # FIXME: move scenario name function to a utils file to avoid code duplication
        num_inputs = len(inputs)
        max_num_digits = math.floor(math.log10(num_inputs)) + 1
        logger.info(f'{self.model.name}: generating input for {num_inputs} scenarios...')
        found_dir = False
        args = []
        for i, input in enumerate(inputs):

            # zero-pad the 1-based scenario index
            num_digits = math.floor(math.log10(i+1)) + 1
            formatted_index = '0' * (max_num_digits - num_digits) + f'{i+1}'
            scenario_name = self.scenario_name.format(index = formatted_index)

            # make the scenario directory if needed
            dir = os.path.join(self.root, scenario_name)
            if os.path.exists(dir):
                found_dir = True
            else:
                print(dir)
                os.mkdir(dir)

            # write input files and define commands
            self.model.write_input_files(input, dir, scenario_name)
            command = self.model.invocation(self.executable, scenario_name)
            dir = os.path.join(self.root, scenario_name)
            args.append({
                'command': command,
                'dir': dir
            })
        if found_dir:
            logger.warning(f'{self.model.name}: one or more existing scenario directories found. Overwriting contents...')
        logger.info(f'{self.model.name}: finished generating scenario input.')

        # now run scenarios in parallel
        pool = multiprocessing.dummy.Pool(self.num_processes)
        logger.info(f'{self.model.name}: running {num_inputs} inputs ({self.num_processes} parallel processes)')

        error_occurred = False

        # this function is called with a list of completed processes by pool.map_async
        def callback(completed_processes) -> None:
            if not all([p.returncode == 0 for p in completed_processes]):
                error_occurred = True

        # this function is called with one of a mapped set of arguments by pool.map_async
        def run_scenario(args) -> subprocess.CompletedProcess:
            f_stdout = open(os.path.join(args['dir'], 'stdout.log'), 'w')
            f_stderr = open(os.path.join(args['dir'], 'stderr.log'), 'w')
            return subprocess.run(args['command'].split(),
                close_fds = True,
                cwd = args['dir'],
                stdout = f_stdout,
                stderr = f_stderr,
            )

        results = pool.map_async(run_scenario, args, callback = callback)
        results.wait()

        logger.info(f'{self.model.name}: completed runs.')
        if error_occurred:
            logger.error(f'{self.model.name}: At least one run failed.')