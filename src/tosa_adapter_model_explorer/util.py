# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
from typing import Any, Dict, Iterable, List

from model_explorer import graph_builder as gb

from . import tosa_1_0


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
    for name in dir(enum):
        if getattr(enum, name) == enum_int:
            return name
    return f"UNKNOWN({enum_int})"


def dict_to_key_value_list(dict: Dict[str, Any]) -> List[gb.KeyValue]:
    """Convert a dictionary to a list of key-value pairs."""
    result = []
    for key, value in dict.items():
        enum_type_name = field_to_enum_map.get(key)
        if enum_type_name and hasattr(tosa_1_0, enum_type_name):
            enum_type = getattr(tosa_1_0, enum_type_name)
            v_str = enum_name(value, enum_type)
        elif isinstance(value, str):
            v_str = value
        elif isinstance(value, bytes):
            v_str = safe_decode(value)
        elif isinstance(value, Iterable):
            v_str = f"[{', '.join(str(v) for v in value)}]"
        else:
            v_str = str(value)
        result.append(gb.KeyValue(key=key, value=v_str))
    return result


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
