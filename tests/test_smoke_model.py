import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.smoke_model import run


class SmokeFailureTests(unittest.TestCase):
    def test_failure_writes_a_machine_readable_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                result = run(Path(directory) / "missing.yaml", output)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result, 1)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["stage"], "dependencies")
            self.assertIn("error_type", record)
            self.assertIn("error", record)


if __name__ == "__main__":
    unittest.main()
