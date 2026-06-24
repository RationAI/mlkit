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
            original_argv = sys.argv[:]
            script_name, user_provided_args = sys.argv[:1], sys.argv[1:]
            sys.argv = script_name + prepend + user_provided_args + append

            try:
                return func(*args, **kwargs)
            finally:
                sys.argv = original_argv

        return wrapper

    return decorator
