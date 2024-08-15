import io
import sys
from collections.abc import Iterable
from types import TracebackType
from typing import Self, TextIO

from rationai.mlkit.stream.stream_logger import StreamLogger
from rationai.mlkit.stream.stream_modifier import StreamModifier


class StreamCapture:
    def __init__(
        self,
        logger: StreamLogger,
        streams: tuple[TextIO, ...] = (sys.stdout, sys.stderr),
    ) -> None:
        self.logger = logger
        self.writer = io.StringIO()
        self.streams_wrapped = [
            StreamModifier(stream, id) for id, stream in enumerate(streams)
        ]

        self.last_writer_id: int | None = None

    def __enter__(self) -> Self:
        self.streams = []
        for stream in self.streams_wrapped:
            stream.set_write(self.write)
            stream.set_writelines(self.writelines)
            stream.set_flush(self.flush)
            self.streams.append(stream)
        return self

    def __exit__(
        self,
        exctype: type[BaseException] | None,
        excinst: BaseException | None,
        exctb: TracebackType | None,
    ) -> None:
        # Synchronize the logger with the last output
        self.logger.log_stream(self.writer.getvalue())

        for stream in self.streams_wrapped:
            stream.teardown()

    def write(self, s: str, stream_id: int) -> None:
        if self.last_writer_id != stream_id:
            self.flush()

            if self.last_writer_id is not None:
                lines = self._get_lines()
                if lines[-1]:
                    lines.append("")
                self._set_writer(lines)

        self.last_writer_id = stream_id

        if s == "\033[A":
            # Jump up
            lines = self._get_lines()
            lines.pop()
            self._set_writer(lines)
        else:
            new_lines = iter(s.split("\n"))

            self._process_line(next(new_lines))

            for line in new_lines:
                self._process_line(line, new=True)

    def writelines(self, lines: Iterable[str]) -> None:
        self.writer.writelines(lines)

    def flush(self) -> None:
        self.logger.log_stream(self.writer.getvalue())

    def _get_lines(self) -> list[str]:
        return self.writer.getvalue().split("\n") or [""]

    def _process_line(self, s: str, new: bool = False) -> None:
        carrages = s.split("\r")
        lines = self._get_lines()

        if new:
            lines.append(carrages[-1])
        else:
            if len(carrages) > 1:
                lines[-1] = carrages[-1]
            else:
                lines[-1] += carrages[-1]

        self._set_writer(lines)

    def _set_writer(self, lines: list[str]) -> None:
        self.writer = io.StringIO("\n".join(lines))
