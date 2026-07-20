import sys
from collections.abc import Callable
from functools import wraps
from typing import Any


def with_cli_args(
    defaults: list[str] | None = None, overrides: list[str] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to injects arguments into sys.argv.

    Args:
        defaults: Arguments injected AFTER script name but BEFORE user args.
            (Acts as defaults: User can override these).
        overrides: Arguments injected AFTER user args.
            (Acts as overrides: Forces value, User cannot override).

    Returns:
        A decorator that modifies sys.argv for the duration of the decorated function.

    Examples:
        >>> from rationai.mlkit import autolog, with_cli_args, MLFlowLogger
        >>> from omegaconf import DictConfig
        >>> import hydra

        >>> @with_cli_args(["+preprocessing=qc"])
        >>> @hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
        >>> @autolog
        >>> def main(config: DictConfig, logger: MLFlowLogger) -> None:
        >>>     pass
    """
    prepend = defaults or []
    append = overrides or []

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 1. Save original state
            original_argv = sys.argv[:]

            # 2. Deconstruct existing argv
            # sys.argv[0] is the script name
            script_name = [sys.argv[0]]
            user_provided_args = sys.argv[1:]

            # 3. Reconstruct: [Script] + [Start] + [User] + [End]
            sys.argv = script_name + prepend + user_provided_args + append

            try:
                return func(*args, **kwargs)
            finally:
                # 4. Restore original state guarantees safety
                sys.argv = original_argv

        return wrapper

    return decorator
