from typing import Any, Optional, List, Dict
from model_explorer import graph_builder as gb
from tosa_1_0 import TosaBasicBlock, Op


def read_file(file_path: str) -> bytes:
    """Read a binary file into byes."""
    with open(file_path, "rb") as file:
        return file.read()


def operator_id(namespace: str, index: int) -> str:
    """Generate a unique operator ID."""
    return f"{namespace}/op{index}"


def op_name(op_val: int) -> str:
    for name in dir(Op):
        if getattr(Op, name) == op_val:
            return name
    return f"UNKNOWN({op_val})"


def find_tensor_in_block(block: TosaBasicBlock, tensor_name: str) -> Optional[Any]:
    """Find a tensor by name in a TosaBasicBlock."""
    tensor_bytes = (
        tensor_name.encode("utf-8") if isinstance(tensor_name, str) else tensor_name
    )

    for i in range(block.TensorsLength()):
        tensor = block.Tensors(i)
        if not tensor:
            continue

        name = tensor.Name()
        if name and name == tensor_bytes:
            return tensor

    return None


def dict_to_key_value_list(attr_map: Dict[str, Any]) -> List[gb.KeyValue]:
    """Convert a dictionary to a list of key-value pairs."""
    result = []
    for k, v in attr_map.items():
        if isinstance(v, list):
            v_str = "[" + ", ".join(str(x) for x in v) + "]"
        else:
            v_str = str(v)
        result.append(gb.KeyValue(key=k, value=v_str))
    return result


def safe_decode(value, default=""):
    """Safely decode bytes to string, handling various input types."""
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)

