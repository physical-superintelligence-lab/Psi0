"""
overwatch.py

Utility class for creating a centralized/standardized logger (built on Rich) and accelerate handler.
"""

import logging
import logging.config
import os
from contextlib import nullcontext
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple, Union

RICH_FORMATTER, DATEFMT = "| >> %(message)s", "[%H:%M:%S %m/%d]"
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "simple-console",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    # "loggers": {
    #     "psi.data.gear.dataset.lerobot": {"level": "DEBUG", "propagate": True},
    #     "psi.data.gear.dataset.lerobot_sharded": {"level": "DEBUG", "propagate": True},
    #     "psi.data.gear.dataset.mixin": {"level": "DEBUG", "propagate": True},
    #     "psi.data.dataset": {"level": "DEBUG", "propagate": True},
    # }
}
logging.config.dictConfig(LOG_CONFIG)

_warned_message = set()


# === Custom Contextual Logging Logic ===
class ContextAdapter(LoggerAdapter):
    CTX_PREFIXES: ClassVar[Dict[int, str]] = {
        **{0: "[*] "},
        **{idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]},
    }

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> Tuple[str, MutableMapping[str, Any]]:
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class DistributedOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that wraps logging with distributed support."""
        self.logger = ContextAdapter(logging.getLogger(name), extra={})
        
        # Get distributed info from environment variables (set by torch.distributed.run or accelerate)
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank = int(os.environ.get("RANK", 0))
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._is_main_process = self._rank == 0

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> only Log `INFO` on Main Process, `ERROR` on others!
        self.logger.setLevel(
            logging.INFO if self._is_main_process else logging.ERROR
        )

    @staticmethod
    def _identity_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Identity decorator that just returns the function unchanged."""
        return fn

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        if self._is_main_process:
            return self._identity_decorator
        else:
            def skip_fn(fn: Callable[..., Any]) -> Callable[..., Any]:
                def wrapper(*args, **kwargs):
                    return None
                return wrapper
            return skip_fn

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        if self._local_rank == 0:
            return self._identity_decorator
        else:
            def skip_fn(fn: Callable[..., Any]) -> Callable[..., Any]:
                def wrapper(*args, **kwargs):
                    return None
                return wrapper
            return skip_fn

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        return nullcontext

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        return nullcontext

    def is_rank_zero(self) -> bool:
        return self._is_main_process

    def rank(self) -> int:
        return self._rank

    def local_rank(self) -> int:
        return self._local_rank

    def world_size(self) -> int:
        return self._world_size

    def warning_once(self, msg):
        if msg not in _warned_message:
            self.warning(msg)
            _warned_message.add(msg)


class PureOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that just wraps logging."""
        self.logger = ContextAdapter(logging.getLogger(name), extra={})

        # Logger Delegation (for convenience; would be nice to just compose & dynamic dispatch eventually)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> INFO
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def get_identity_ctx() -> Callable[..., Any]:
        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return identity

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return self.get_identity_ctx()

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        return self.get_identity_ctx()

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        return nullcontext

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        return nullcontext

    @staticmethod
    def is_rank_zero() -> bool:
        return True

    @staticmethod
    def rank() -> int:
        return 0

    def local_rank(self) -> int:
        return 0

    @staticmethod
    def world_size() -> int:
        return 1

    def warning_once(self, msg):
        if msg not in _warned_message:
            self.warning(msg)
            _warned_message.add(msg)


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    return (
        DistributedOverwatch(name)
        if int(os.environ.get("WORLD_SIZE", -1)) != -1
        else PureOverwatch(name)
    )
