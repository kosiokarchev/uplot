import typing as tp
from abc import ABC
from functools import partial
from itertools import chain
from operator import is_not


class Orderable:
    def __init__(self, *args, zorder=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.zorder = zorder


class Printable(Orderable):
    _header: str = None
    _footer: str = None
    _joiner: str = '\n'

    def __init__(self, *args, zorder=0, children=None, _header=None, _footer=None, _joiner=None, **kwargs):
        super().__init__(*args, zorder=zorder, **kwargs)
        self.children: tp.MutableSequence[tp.Union[Printable, Command, str]] = [] if children is None else children
        if _header is not None:
            self._header = _header
        if _footer is not None:
            self._footer = _footer
        if _joiner is not None:
            self._joiner = _joiner

    @property
    def header(self) -> tp.Optional[str]:
        return self._header

    @property
    def body(self):
        return ('\t' + c
                for ch in sorted(self.children, key=self._sort_key)
                for c in (ch.print() if isinstance(ch, Printable)
                          else str(ch).split('\n')))

    @property
    def footer(self) -> tp.Optional[str]:
        return self._footer

    @staticmethod
    def _sort_key(printable_or_string):
        return printable_or_string.zorder if isinstance(printable_or_string, Orderable) else 0

    def print(self) -> tp.Iterable[str]:
        return filter(partial(is_not, None), chain(
            (self.header,),
            self.body,
            (self.footer,)
        ))

    def __str__(self):
        return self._joiner.join(self.print())


class Optionable:
    no_output = object()
    defaults = {}

    _OptionsType = tp.Union[tp.MutableMapping[str, tp.Union[str, tp.Any]], tp.Iterable[str]]

    def __init__(self, *args, options: _OptionsType = None, additional_options='', **kwargs):
        super().__init__(*args, **kwargs)

        self.options: Optionable._OptionsType = (
            options if isinstance(options, tp.Mapping)
            else {key: None for key in options} if options else {}
        )
        self.additional_options = additional_options

    def _set_options(self, delete_if_None=True, **kwargs):
        for key, val in kwargs.items():
            if val is None:
                if delete_if_None:
                    self.options.pop(key, None)
            else:
                self.options[key] = val

    @staticmethod
    def optional(arg: str):
        return arg and f'[{arg}]'

    @classmethod
    def format_options(cls, additional_options: str = None, **options: str):
        return ', '.join(filter(bool, (', '.join(cls.format_options_iter(**options)), additional_options)))

    @classmethod
    def format_options_iter(cls, **options: str) -> tp.Iterable[str]:
        return (f'{key}={{{val}}}' if val is not None else key
                for key, val in options.items()
                if val not in (Optionable.no_output, cls.defaults.get(key, Optionable.no_output)))

    @property
    def formatted_options(self):
        return self.format_options(self.additional_options, **self.options)

    def __str__(self):
        return self.formatted_options


class PrintableEnvironment(Printable, Optionable, ABC):
    name: str = None

    @property
    def header(self):
        return rf'\begin{{{self.name}}}{self.optional(self.formatted_options)}'

    @property
    def footer(self):
        return rf'\end{{{self.name}}}'


class Command(Optionable, Orderable, ABC):
    name: str

    def __init__(self, command_body='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_body = command_body

    @property
    def formatted_name(self):
        return f'\\{self.name}'

    @property
    def formatted_command_body(self):
        return self.command_body or ''

    @property
    def end(self):
        return ';'

    def __str__(self):
        return f'{self.formatted_name}{self.optional(self.formatted_options)}{self.formatted_command_body}{self.end}'
