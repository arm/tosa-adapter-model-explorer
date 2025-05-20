from typing import Any, List, Dict
from types import ModuleType
from model_explorer import graph_builder as gb


def read_file(file_path: str) -> bytes:
    """Read a binary file into byes."""
    with open(file_path, "rb") as file:
        return file.read()


def operator_id(namespace: str, index: int) -> str:
    """Generate a unique operator ID."""
    return f"{namespace}/op{index}"


def enum_name(enum_int: int, enum: Any) -> str:
    for name in dir(enum):
        if getattr(enum, name) == enum_int:
            return name
    return f"UNKNOWN({enum_int})"


def dict_to_key_value_list(
    dict: Dict[str, Any], tosa_module: ModuleType
) -> List[gb.KeyValue]:
    """Convert a dictionary to a list of key-value pairs."""
    result = []
    for key, value in dict.items():
        enum_type_name = field_to_enum_map.get(key)
        if enum_type_name and hasattr(tosa_module, enum_type_name):
            enum_type = getattr(tosa_module, enum_type_name)
            v_str = enum_name(value, enum_type)
        elif isinstance(value, bytes):
            v_str = safe_decode(value)
        else:
            v_str = str(value)
        result.append(gb.KeyValue(key=key, value=v_str))
    return result


def safe_decode(value, default=""):
    """Safely decode bytes to string, handling various input types."""
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
