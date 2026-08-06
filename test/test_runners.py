import tempfile
import unittest

import ambrs.aerosol as aerosol
import ambrs.partmc as partmc
import ambrs.runners as runners


def make_model():
    return partmc.AerosolModel(
        processes=aerosol.AerosolProcesses(aging=True, coagulation=True),
        n_part=1,
    )


class TestPoolRunner(unittest.TestCase):
    def test_constructor_succeeds_with_temp_root_and_sets_defaults(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = runners.PoolRunner(
                model=model,
                executable="/bin/echo",
                root=temp_dir,
            )

            self.assertIs(runner.model, model)
            self.assertEqual(runner.executable, "/bin/echo")
            self.assertEqual(runner.root, temp_dir)
            self.assertGreaterEqual(runner.num_processes, 1)
            self.assertEqual(runner.scenario_name, "{index}")
            self.assertEqual(runner.scenario_start, 1)
            self.assertIsNone(runner.max_num_digits)

    def test_constructor_raises_for_missing_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = f"{temp_dir}/missing"

            with self.assertRaisesRegex(OSError, "root path"):
                runners.PoolRunner(
                    model=make_model(),
                    executable="/bin/echo",
                    root=missing_root,
                )

    def test_constructor_raises_for_invalid_scenario_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "scenario_start"):
                runners.PoolRunner(
                    model=make_model(),
                    executable="/bin/echo",
                    root=temp_dir,
                    scenario_start=0,
                )

    def test_run_returns_empty_list_for_empty_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = runners.PoolRunner(
                model=make_model(),
                executable="/bin/echo",
                root=temp_dir,
            )

            self.assertEqual(runner.run([]), [])

    def test_run_rejects_non_list_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = runners.PoolRunner(
                model=make_model(),
                executable="/bin/echo",
                root=temp_dir,
            )

            with self.assertRaisesRegex(TypeError, "inputs must be a list"):
                runner.run(())


if __name__ == "__main__":
    unittest.main()
