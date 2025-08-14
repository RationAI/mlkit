import io
import re
import sys
import traceback
from collections.abc import Callable, Iterable
from functools import partial
from types import TracebackType
from typing import Self, TextIO
from unittest.mock import patch

from rationai.mlkit.stream.stream_logger import StreamLogger


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
                write=partial(self._write, stream_write=stream.write),
                writelines=partial(
                    self._writelines, stream_writelines=stream.writelines
                ),
                flush=partial(self._flush, stream_flush=stream.flush),
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

        self.logger.log_stream(self._buffer.getvalue())

    def _write(self, s: str, stream_write: Callable[[str], int]) -> int:
        result = stream_write(s)

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
        self.logger.log_stream(self._buffer.getvalue())
        return result

    def _writelines(
        self, lines: Iterable[str], stream_writelines: Callable[[Iterable[str]], None]
    ) -> None:
        stream_writelines(lines)
        self._buffer.writelines(lines)

    def _flush(self, stream_flush: Callable[[], None]) -> None:
        stream_flush()
        self.logger.log_stream(self._buffer.getvalue())
