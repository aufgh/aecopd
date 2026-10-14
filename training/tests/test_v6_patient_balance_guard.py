import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from training.v6 import patient_balance_audit,patient_balance_finish


class Tests(unittest.TestCase):
    def test_audit_and_finish_reject_running_training_before_data_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'.running.lock').touch()
            for module,entry in ((patient_balance_audit,patient_balance_audit.verify),
                (patient_balance_finish,patient_balance_finish.run)):
                with patch.object(module,'ROOT',root):
                    with self.assertRaisesRegex(RuntimeError,'running'):entry()


if __name__=='__main__':unittest.main()
