from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any


class DatasetFilter:
    def __init__(self, filter_lambdas: Sequence[str] | None = None) -> None:
        self.filter_lambdas = list(filter_lambdas or [])
        self.filters = []
        for expr in self.filter_lambdas:
            try:
                predicate = eval(expr)
            except Exception as exc:
                print(f"Invalid filter: {expr}", file=sys.stderr)
                raise ValueError(f"Invalid filter: {expr}") from exc
            if not callable(predicate):
                print(f"Invalid filter: {expr}", file=sys.stderr)
                raise ValueError(f"Invalid filter: {expr}")
            self.filters.append(predicate)

    def __repr__(self) -> str:
        return f"DatasetFilter(filter_lambdas={self.filter_lambdas!r})"

    def matches(self, row: Mapping[str, Any]) -> bool:
        row = dict(row)
        if row.get("is_deleted", False):
            return False
        for expr, predicate in zip(self.filter_lambdas, self.filters, strict=True):
            result = predicate(row)
            if not isinstance(result, bool):
                raise TypeError(f"Filter must return bool: {expr}")
            if not result:
                return False
        return True
