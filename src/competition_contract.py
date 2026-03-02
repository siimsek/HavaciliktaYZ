"""Competition runtime contract: typed errors and deterministic error decisions."""

from dataclasses import dataclass
from enum import Enum


class CompetitionError(Exception):
    """Base class for competition runtime errors."""


class RecoverableIOError(CompetitionError):
    """Transient IO/runtime errors that may recover with retry."""


class DataContractError(CompetitionError):
    """Schema or data-shape violations that require degrade or stop."""


class FatalSystemError(CompetitionError):
    """Fatal process/runtime failures that should stop the session."""


class ErrorDecision(str, Enum):
    RETRY = "RETRY"
    DEGRADE = "DEGRADE"
    STOP = "STOP"


@dataclass
class ErrorPolicy:
    """Centralized, deterministic error decision policy."""

    retry_budget: int = 8
    degrade_budget: int = 3
    recoverable_count: int = 0
    contract_count: int = 0
    unknown_count: int = 0

    def decide_on_error(self, error: Exception) -> ErrorDecision:
        if isinstance(error, FatalSystemError):
            return ErrorDecision.STOP

        if isinstance(error, DataContractError):
            self.contract_count += 1
            if self.contract_count > self.degrade_budget:
                return ErrorDecision.STOP
            return ErrorDecision.DEGRADE

        if isinstance(error, RecoverableIOError):
            self.recoverable_count += 1
            if self.recoverable_count > self.retry_budget:
                return ErrorDecision.DEGRADE
            return ErrorDecision.RETRY

        # Unknown errors are never treated as directly recoverable.
        self.unknown_count += 1
        if self.unknown_count > self.degrade_budget:
            return ErrorDecision.STOP
        return ErrorDecision.DEGRADE

