import io
import re
import sys
import traceback
from collections.abc import Iterable
from functools import wraps
from types import TracebackType
from typing import Self, TextIO
from unittest.mock import patch

from rationai.mlkit.stream.stream_logger import StreamLogger


def create_wrapper(original_method, custom_handler):
    @wraps(original_method)
    def wrapper(*args, **kwargs):
        result = original_method(*args, **kwargs)
        custom_handler(*args, **kwargs)
        return result

    return wrapper


class StreamCapture:
    ANSI_ESCAPE_SEQ = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(
        self, logger: StreamLogger, streams: Iterable[TextIO] = (sys.stdout, sys.stderr)
    ) -> None:
        self.logger = logger
        self._buffer = io.StringIO()

        self._patchers = [
            patch.multiple(
                stream,
                write=create_wrapper(stream.write, self.write),
                writelines=create_wrapper(stream.writelines, self.writelines),
                flush=create_wrapper(stream.flush, self.flush),
            )
            for stream in streams
        ]

    def __enter__(self) -> Self:
        for patcher in self._patchers:
            patcher.start()

        return self

    def __exit__(
        self,
        exctype: type[BaseException] | None,
        excinst: BaseException | None,
        exctb: TracebackType | None,
    ) -> None:
        for patcher in self._patchers:
            patcher.stop()

        if exctype is not None:
            traceback_str = "".join(traceback.format_exception(exctype, excinst, exctb))
            self._buffer.write(traceback_str)

        self.flush()

    def write(self, s: str) -> None:
        s = self.ANSI_ESCAPE_SEQ.sub("", s)

        # # Move to the end of the buffer to write new content
        self._buffer.seek(0, io.SEEK_END)

        if "\r" in s:
            # If there's a carriage return, we might need to overwrite a line
            self._buffer.seek(0)
            content = self._buffer.read()
            last_newline = content.rfind("\n")

            # Position the cursor at the beginning of the last line
            self._buffer.seek(last_newline + 1)
            self._buffer.truncate()  # Clear the last line

            # Write the part of the string that comes after the last CR
            s = s.rsplit("\r", 1)[-1]

        self._buffer.write(s)
        self.flush()

    def writelines(self, lines: Iterable[str]) -> None:
        self._buffer.writelines(lines)

    def flush(self) -> None:
        self.logger.log_stream(self._buffer.getvalue())
