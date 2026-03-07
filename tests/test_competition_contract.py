import unittest

from src.competition_contract import (
    DataContractError,
    ErrorDecision,
    ErrorPolicy,
    FatalSystemError,
    RecoverableIOError,
)

class TestErrorPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = ErrorPolicy(retry_budget=2, degrade_budget=1)

    def test_fatal_system_error_stops(self):
        decision = self.policy.decide_on_error(FatalSystemError())
        self.assertEqual(decision, ErrorDecision.STOP)

    def test_data_contract_error_degrades_then_stops(self):
        # 1st time -> DEGRADE
        self.assertEqual(self.policy.decide_on_error(DataContractError()), ErrorDecision.DEGRADE)
        # 2nd time (exceeds budget 1) -> STOP
        self.assertEqual(self.policy.decide_on_error(DataContractError()), ErrorDecision.STOP)
        # 3rd time -> STOP
        self.assertEqual(self.policy.decide_on_error(DataContractError()), ErrorDecision.STOP)

    def test_recoverable_io_error_retries_then_degrades(self):
        # 1st time -> RETRY
        self.assertEqual(self.policy.decide_on_error(RecoverableIOError()), ErrorDecision.RETRY)
        # 2nd time -> RETRY
        self.assertEqual(self.policy.decide_on_error(RecoverableIOError()), ErrorDecision.RETRY)
        # 3rd time (exceeds budget 2) -> DEGRADE
        self.assertEqual(self.policy.decide_on_error(RecoverableIOError()), ErrorDecision.DEGRADE)

    def test_unknown_error_degrades_then_stops(self):
        # 1st time -> DEGRADE
        self.assertEqual(self.policy.decide_on_error(ValueError("Unknown")), ErrorDecision.DEGRADE)
        # 2nd time (exceeds budget 1) -> STOP
        self.assertEqual(self.policy.decide_on_error(TypeError("Unknown 2")), ErrorDecision.STOP)
        # 3rd time -> STOP
        self.assertEqual(self.policy.decide_on_error(KeyError("Unknown 3")), ErrorDecision.STOP)

if __name__ == "__main__":
    unittest.main()
