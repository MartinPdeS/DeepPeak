"""Common DeepPeak exception types."""


class DeepPeakError(Exception):
    """Base exception for expected DeepPeak errors."""


class InvalidTraceError(DeepPeakError, ValueError):
    """Raised when a trace cannot be represented or processed."""


class InvalidConfigurationError(DeepPeakError, ValueError):
    """Raised when workflow settings are invalid or incompatible."""


class InvalidDetectorError(InvalidConfigurationError):
    """Raised when a detector name or detector combination is unsupported."""


class MissingDetectorError(DeepPeakError, TypeError):
    """Raised when a workflow is created without a required detector."""


class AnalysisStateError(DeepPeakError, RuntimeError):
    """Raised when an analysis operation requires a missing prior result."""


class AnalysisInputError(DeepPeakError, ValueError):
    """Raised when analysis input cannot be processed."""


class MissingOptionalDependencyError(DeepPeakError, ImportError):
    """Raised when an optional subsystem dependency is unavailable."""
