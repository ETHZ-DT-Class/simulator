from enum import IntEnum
from typing import Dict, Iterable, List, Set, Union
from termcolor import colored


class LogLevel(IntEnum):
    DEBUG_TIMING = 10
    DEBUG = 20
    INFO = 30
    WARNING = 40
    ERROR = 50
    CRITICAL = 60
    SUCCESS = 70


class Logger:
    pass


class Logger:

    name: str = ""
    level: LogLevel = LogLevel.INFO
    prefix_txt: str = ""
    active_loggers: Dict[str, Logger] = dict()

    def __init__(self, name="", level=LogLevel.INFO):
        self.name = name
        self.set_name(name)
        self.set_level(level)
        self.set_prefix_txt("")

    def set_name(self, name: str) -> None:
        self.name = name

    def set_level(self, level: LogLevel) -> None:

        if not isinstance(level, LogLevel):
            done = False
            if isinstance(level, str):
                try:
                    level = LogLevel[level.upper()]
                    done = True
                except:
                    pass
            if not done:
                txt_error = (
                    f"Invalid logger level: {self.bold_red_on_black(level)}. Must be one of"
                    + f" {self.bold_green_on_black([lvl.lower() for lvl in LogLevel.__members__.keys()])}"
                )
                self.level = LogLevel.INFO
                self.error(txt_error)
                raise ValueError(self.red(txt_error))

        self.level = level

    def set_prefix_txt(self, prefix_txt: str) -> None:
        self.prefix_txt = prefix_txt

    def debug_timing(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.DEBUG_TIMING:
            return
        print(self._log_formatter(LogLevel.DEBUG_TIMING, txt), *args, **kwargs)

    def debug(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.DEBUG:
            return
        print(self._log_formatter(LogLevel.DEBUG, txt), *args, **kwargs)

    def info(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.INFO:
            return
        print(self._log_formatter(LogLevel.INFO, txt), *args, **kwargs)

    def warning(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.WARNING:
            return
        print(self._log_formatter(LogLevel.WARNING, txt), *args, **kwargs)
        
    def warning_whole(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.WARNING:
            return
        print(self._log_formatter(LogLevel.WARNING, txt, whole=True), *args, **kwargs)

    def error(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.ERROR:
            return
        print(self._log_formatter(LogLevel.ERROR, txt), *args, **kwargs)

    def error_whole(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.ERROR:
            return
        print(self._log_formatter(LogLevel.ERROR, txt, whole=True), *args, **kwargs)

    def critical(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.CRITICAL:
            return
        print(self._log_formatter(LogLevel.CRITICAL, txt), *args, **kwargs)

    def success(self, txt: str, *args, **kwargs) -> None:
        if self.level > LogLevel.SUCCESS:
            return
        print(self._log_formatter(LogLevel.SUCCESS, txt), *args, **kwargs)

    @staticmethod
    def str_formatter(
        txt: Union[str, Iterable[str]], color=None, on_color=None, attrs=None
    ) -> str:
        if isinstance(txt, (str, bool, int, float)):
            return colored(txt, color, on_color, attrs)
        elif isinstance(txt, Iterable):
            return "".join(
                ["[", ", ".join([colored(t, color, on_color, attrs) for t in txt]), "]"]
            )
        else:
            # print(
            #     Logger.str_formatter(
            #         f"logger meta-warning: not fully supported input type {type(txt)} for variable '{txt}'",
            #         "yellow",
            #     )
            # )
            return colored(txt, color, on_color, attrs)

    @classmethod
    def red(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, "red")

    @classmethod
    def yellow(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, "yellow")

    @classmethod
    def green(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, "green")

    @classmethod
    def bold(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, attrs=["bold"])

    @classmethod
    def bold_green_on_black(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, "green", "on_grey", ["bold"])

    @classmethod
    def bold_yellow_on_black(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, "yellow", "on_grey", ["bold"])

    @classmethod
    def bold_red_on_black(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, "red", "on_grey", ["bold"])

    @classmethod
    def blink_green(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, "green", attrs=["blink"])

    @classmethod
    def blink_red(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, "red", attrs=["blink"])

    @classmethod
    def dark(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, attrs=["dark"])

    @classmethod
    def concealed(cls, txt: Union[str, Iterable[str]]) -> str:
        return cls.str_formatter(txt, attrs=["concealed"])

    def underline(self, txt: Union[str, Iterable[str]]) -> str:
        return self.str_formatter(txt, attrs=["underline"])

    def _log_formatter(self, log_level: LogLevel, txt: str, whole: bool = False) -> str:

        log_level_name = log_level.name
        prefix_txt = self.prefix_txt
        color = None
        on_color = None
        attrs = None

        if log_level == LogLevel.DEBUG_TIMING:
            color = "blue"
        elif log_level == LogLevel.DEBUG:
            color = "blue"
        elif log_level == LogLevel.INFO:
            ...
        elif log_level == LogLevel.WARNING:
            color = "yellow"
        elif log_level == LogLevel.ERROR:
            color = "red"
            attrs = ["blink", "underline"]
        elif log_level == LogLevel.CRITICAL:
            color = "magenta"
            attrs = ["blink", "underline"]
        elif log_level == LogLevel.SUCCESS:
            color = "green"
            attrs = ["blink"]

        colon = ": " if self.name else ""
        if whole:
            return self.str_formatter(
                f"[{self.name}{colon}{log_level_name}] {prefix_txt}{txt}",
                color,
                on_color,
                attrs,
            )
        else:
            return f"{self.str_formatter(f'[{self.name}{colon}{log_level_name}]', color, on_color, attrs)} {prefix_txt}{txt}"
