# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
import struct
from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Sized,
    TypeVar,
)

import numpy as np
from model_explorer import graph_builder as gb

from . import tosa_1_0

# Pre-built reverse lookup tables for enum classes (built once at module load)
_ENUM_LOOKUPS: Dict[Any, Dict[int, str]] = {}
_DTYPE_STRUCT_FORMATS = {
    "BOOL": (1, "?"),
    "INT8": (1, "b"),
    "INT16": (2, "h"),
    "INT32": (4, "i"),
    "FP16": (2, "e"),
    "FP32": (4, "f"),
    "SHAPE": (8, "q"),
}


class _DecodedDataPreview(Sized, Iterable[Any]):
    """Iterable preview that fakes the total len while only containing a fraction of the elements."""

    def __init__(self, values: list[Any], total_count: int):
        self._values = values
        self._total_count = total_count

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return self._total_count


def _get_enum_lookup(enum_class: Any) -> Dict[int, str]:
    """Build reverse lookup table for an enum class (cached)."""
    if enum_class not in _ENUM_LOOKUPS:
        _ENUM_LOOKUPS[enum_class] = {
            v: k
            for k, v in vars(enum_class).items()
            if isinstance(v, int) and not k.startswith("_")
        }
    return _ENUM_LOOKUPS[enum_class]


def read_file(file_path: str) -> bytes:
    """Read a binary file into bytes.

    Args:
        file_path: Path to the file to read.

    Returns:
        The contents of the file as bytes.
    """
    with open(file_path, "rb") as file:
        return file.read()


def operator_id(namespace: str, index: int) -> str:
    """Generate a unique operator ID within a namespace.

    Args:
        namespace: Namespace identifier for the operator.
        index: Index of the operator within the namespace.

    Returns:
        A string representing the unique operator ID.
    """
    return f"{namespace}/op{index}"


def enum_name(enum_int: int, enum: Any) -> str:
    """Get the name of an enum value using cached reverse lookup."""
    lookup = _get_enum_lookup(enum)
    return lookup.get(enum_int, f"UNKNOWN({enum_int})")


def dict_to_key_value_list(
    dict: Dict[str, Any], max_array_elements: int, is_shape: bool = False
) -> List[gb.KeyValue]:
    """Convert a dictionary to a list of key-value pairs."""
    result = []
    for key, value in dict.items():
        enum_type_name = field_to_enum_map.get(key)

        if key == "data":
            if value is None:
                v_str = str(value)
            else:
                dtype = _try_get_dtype(dict)
                if dtype is None and is_shape:
                    dtype = "SHAPE"
                v_str = _stringify_array(
                    _decode_data(
                        value,
                        dtype,
                        max_elements=max_array_elements + 1,
                    ),
                    max_array_elements,
                )
        elif enum_type_name and hasattr(tosa_1_0, enum_type_name):
            enum_type = getattr(tosa_1_0, enum_type_name)
            v_str = enum_name(value, enum_type)
        elif isinstance(value, str):
            v_str = value
        elif isinstance(value, bytes):
            v_str = safe_decode(value)
        elif isinstance(value, Iterable):
            v_str = _stringify_array(value, max_array_elements)
        else:
            v_str = str(value)
        result.append(gb.KeyValue(key=key, value=v_str))
    return result


def _try_get_dtype(dict: Dict[str, Any]) -> str | None:
    dtype = None
    if hasattr(tosa_1_0, "DType"):
        enum_type = tosa_1_0.DType
        type_name = dict.get("type")
        if type_name is not None:
            dtype = enum_name(type_name, enum_type)
    return dtype


def _decode_data(
    data: bytes | Sequence[int] | np.ndarray | None,
    dtype: str | None,
    max_elements: int | None = None,
) -> Iterable[Any] | _DecodedDataPreview:
    if data is None:
        return []

    raw = _raw_data_to_bytes(data)
    if raw is None:
        return data

    if dtype == "INT48":
        return _unpack_int48_values(raw, max_elements=max_elements)

    format_entry = _DTYPE_STRUCT_FORMATS.get(dtype or "")
    if format_entry is not None:
        dtype_size, dtype_fmt = format_entry
        return _unpack_values(
            raw,
            dtype_size=dtype_size,
            dtype_fmt=dtype_fmt,
            max_elements=max_elements,
        )

    return raw


def _raw_data_to_bytes(
    data: bytes | Sequence[int] | np.ndarray,
) -> bytes | None:
    if isinstance(data, bytes):
        return data
    if isinstance(data, np.ndarray):
        return data.tobytes()

    try:
        return bytes(data)
    except (TypeError, ValueError):
        return None


def _unpack_values(
    raw: bytes,
    dtype_size: int,
    dtype_fmt: str,
    max_elements: int | None = None,
) -> list[Any] | _DecodedDataPreview:
    """Decode all complete little-endian values from bytes."""
    byte_count = len(raw) - (len(raw) % dtype_size)
    element_count = byte_count // dtype_size
    if element_count == 0:
        return []

    decode_count = _decode_count(element_count, max_elements)
    values = list(
        struct.unpack(
            f"<{decode_count}{dtype_fmt}", raw[: decode_count * dtype_size]
        )
    )

    if max_elements is None:
        return values
    return _DecodedDataPreview(values, element_count)


def _unpack_int48_values(
    raw: bytes, max_elements: int | None = None
) -> list[int] | _DecodedDataPreview:
    """Decode all complete little-endian signed 48-bit values from bytes."""
    dtype_size = 6
    byte_count = len(raw) - (len(raw) % dtype_size)
    element_count = byte_count // dtype_size
    decode_count = _decode_count(element_count, max_elements)
    values = [
        int.from_bytes(raw[idx : idx + dtype_size], "little", signed=True)
        for idx in range(0, decode_count * dtype_size, dtype_size)
    ]

    if max_elements is None:
        return values
    return _DecodedDataPreview(values, element_count)


def _decode_count(element_count: int, max_elements: int | None) -> int:
    if max_elements is None:
        return element_count
    return max(0, min(element_count, max_elements))


def _stringify_array(value: Iterable[Any], max_array_elements: int) -> str:
    """Convert an iterable to a compact string, truncating without full copies."""
    n = len(value) if isinstance(value, Sized) else None
    value_list = list(islice(value, max_array_elements + 1))
    preview = value_list[:max_array_elements]
    preview_str = ", ".join(map(str, preview))

    if n is None or n > max_array_elements:
        n_str = str(n) if n is not None else "unknown number of"
        return f"(showing {max_array_elements} out of {n_str} elements)\n[{preview_str}...]"

    return f"[{preview_str}]"


def safe_decode(value: Any, default: str = "") -> str:
    """Safely decode a value to a string.

    Handles bytes, None, and other types.

    Args:
        value: The value to decode.
        default: Default string if value is None.

    Returns:
        Decoded string or default.
    """
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


field_to_enum_map = {
    "type": "DType",
    "accType": "DType",
    "accumDtype": "DType",
    "mode": "ResizeMode",
    "nanMode": "NanPropagationMode",
}

T = TypeVar("T")


def iter_vector(getter: Callable[[int], T | None], length: int) -> Iterable[T]:
    """Iterate over elements in a FlatBuffers vector."""
    for idx in range(length):
        item = getter(idx)
        if item:
            yield item
