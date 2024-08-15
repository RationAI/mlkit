from abc import ABC, abstractmethod


class StreamLogger(ABC):
    @abstractmethod
    def log_stream(self, text: str) -> None: ...
