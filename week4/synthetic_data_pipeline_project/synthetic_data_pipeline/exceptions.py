# exceptions.py
class PipelineError(Exception):
    """Base class for pipeline errors."""
    pass

class InvalidPathError(PipelineError):
    """Raised when a file path is invalid or inaccessible."""
    pass

class DataGenerationError(PipelineError):
    """Raised when data generation fails due to wrong parameters."""
    pass
