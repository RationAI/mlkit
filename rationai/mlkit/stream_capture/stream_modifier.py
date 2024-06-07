from collections.abc import Callable, Iterable
from typing import Any, TextIO, TypeVar


T = TypeVar("T")
A = TypeVar("A")


class StreamModifier:
    def __init__(self, stream: TextIO, id: int) -> None:
        self.stream = stream
        self.id = id
        self.originals: dict[str, Callable[...]] = {}

    def __del__(self) -> None:
        self.teardown()

    def set_write(self, write: Callable[[str, int], None]) -> None:
        self._set_method("write", lambda s: write(s, self.id))

    def set_writelines(self, writelines: Callable[[Iterable[str]], None]) -> None:
        self._set_method("writelines", writelines)

    def set_flush(self, flush: Callable[[], None]) -> None:
        self._set_method("flush", flush)

    def reset_write(self) -> None:
        self._reset_method("write")

    def reset_writelines(self) -> None:
        self._reset_method("writelines")

    def reset_flush(self) -> None:
        self._reset_method("flush")

    def teardown(self) -> None:
        self.reset_write()
        self.reset_writelines()
        self.reset_flush()

    def _set_method(self, method_name: str, method: Callable[..., None]) -> None:
        self._reset_method(method_name)
        original_method = getattr(self.stream, method_name)

        def _method(*args: Any, **kwargs: Any) -> Any:
            method(*args, **kwargs)
            return original_method(*args, **kwargs)

        setattr(self.stream, method_name, _method)
        self.originals[method_name] = original_method

    def _reset_method(self, method_name: str) -> None:
        if method_name in self.originals:
            setattr(self.stream, method_name, self.originals[method_name])
            del self.originals[method_name]
