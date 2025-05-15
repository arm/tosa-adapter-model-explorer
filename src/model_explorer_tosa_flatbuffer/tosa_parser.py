from pathlib import Path
from types import ModuleType
from typing import List, Dict, Callable, Any

from tosa_1_0 import TosaGraph, TosaOperator, Op, TosaBasicBlock
from .util import (
    read_file,
    find_tensor_in_block,
    operator_id,
    op_name,
    dict_to_key_value_list,
    safe_decode,
)
import tosa_0_8
import tosa_1_0

from model_explorer import graph_builder as gb


class TosaParser:
    """
    A parser for TOSA flatbuffer files that handles multiple schema versions.

    This class parses TOSA flatbuffer files with schema versions 0.8 and 1.0,
    correctly handling the differences in attribute structures between versions.
    """

    def __init__(self, file_path: str):
        """
        Initialize the parser with a file path.

        Args:
            file_path: Path to the TOSA flatbuffer file.
        """
        self.file_path = file_path
        self.buffer = read_file(file_path)
        self.root_graph = TosaGraph.GetRootAsTosaGraph(self.buffer, 0)
        self.tosa_module = self._get_tosa_module()

    def _get_tosa_module(self) -> ModuleType:
        """Get the correct TOSA module for the flatbuffer file version."""
        version_obj = self.root_graph.Version()
        if not version_obj:
            return tosa_1_0

        tosa_version = (version_obj._Major(), version_obj._Minor())

        return tosa_0_8 if tosa_version[0] < 1 else tosa_1_0

    def parse(self) -> gb.GraphCollection:
        """
        Parse the TOSA file into a GraphCollection.

        Returns:
            GraphCollection containing the parsed graph.
        """
        graphs: List[gb.Graph] = []

        for region_idx in range(self.root_graph.RegionsLength()):
            region = self.root_graph.Regions(region_idx)
            if not region:
                continue

            for block_idx in range(region.BlocksLength()):
                block = region.Blocks(block_idx)
                if not block:
                    continue

                graph_id = safe_decode(block.Name(), f"block{block_idx}")
                graphs.append(self._build_graph(block, graph_id))

        return gb.GraphCollection(label=Path(self.file_path).stem, graphs=graphs)

    def _build_graph(self, block: TosaBasicBlock, graph_id: str) -> gb.Graph:
        """
        Build a graph from a TOSA basic block.

        Args:
            block: The TOSA basic block to process.
            graph_id: Identifier for the graph.

        Returns:
            Graph object.
        """
        nodes: Dict[str, gb.GraphNode] = {}

        tensor_producer_map = self._map_outputs(block, graph_id)
        self._add_nodes(nodes, block, graph_id, tensor_producer_map)

        return gb.Graph(id=graph_id, nodes=list(nodes.values()))

    def _map_outputs(self, block: TosaBasicBlock, namespace: str) -> Dict[str, str]:
        """
        Map tensor names to the operators that produce them.

        Args:
            block: The TOSA basic block.
            namespace: Namespace for the node IDs.

        Returns:
            Dictionary mapping tensor names to producer node IDs.
        """
        output_map: Dict[str, str] = {}

        for i in range(block.OperatorsLength()):
            op = block.Operators(i)
            if op and op.Op() != Op.CONST:
                op_id = operator_id(namespace, i)

                for j in range(op.OutputsLength()):
                    output_name = safe_decode(op.Outputs(j))
                    output_map[output_name] = op_id

        return output_map

    def _add_nodes(
        self,
        nodes: Dict[str, gb.GraphNode],
        block: TosaBasicBlock,
        namespace: str,
        tensor_producer_map: Dict[str, str],
    ) -> None:
        """
        Add nodes to the graph from TOSA operators.

        Args:
            nodes: Dictionary to add nodes to.
            block: The TOSA basic block.
            namespace: Namespace for the node IDs.
            tensor_producer_map: Mapping of tensor names to producer nodes.
        """
        for i in range(block.OperatorsLength()):
            operator = block.Operators(i)
            if not operator or operator.Op() == Op.CONST:
                continue

            node_id = operator_id(namespace, i)
            attr_dict = self._collect_operator_attrs(operator)

            nodes[node_id] = gb.GraphNode(
                id=node_id,
                label=op_name(operator.Op()),
                namespace=namespace,
                incomingEdges=self._add_incoming_edges(operator, tensor_producer_map),
                attrs=dict_to_key_value_list(attr_dict),
                inputsMetadata=self._collect_operator_metadata(
                    block, operator.Inputs, operator.InputsLength
                ),
                outputsMetadata=self._collect_operator_metadata(
                    block, operator.Outputs, operator.OutputsLength
                ),
            )

    def _add_incoming_edges(
        self, operator: TosaOperator, tensor_producer_map: Dict[str, str]
    ) -> List[gb.IncomingEdge]:
        """
        Add incoming edges to a node.

        Args:
            operator: The TOSA operator.
            tensor_producer_map: Mapping of tensor names to producer nodes.

        Returns:
            List of IncomingEdge objects.
        """
        incoming_edges: List[gb.IncomingEdge] = []

        for input_idx in range(operator.InputsLength()):
            input_tensor_name = safe_decode(operator.Inputs(input_idx))
            source_node_id = tensor_producer_map.get(input_tensor_name)

            if source_node_id:
                incoming_edges.append(
                    gb.IncomingEdge(
                        sourceNodeId=source_node_id,
                        sourceNodeOutputId=safe_decode(operator.Outputs(0)),
                        targetNodeInputId=input_tensor_name,
                    )
                )

        return incoming_edges

    def _collect_operator_attrs(
        self,
        op: TosaOperator,
    ) -> Dict[str, Any]:
        """
        Collect attributes from a tosa operator using the appropriate version module.

        Args:
        op: The TOSA operator containing the attribute

        Returns: Dictionary of parsed attribute fields
        """

        attr_type = op.AttributeType()
        attr_table = op.Attribute()

        if attr_type == 0 or not attr_table:
            return {}

        attr_obj = self.tosa_module.AttributeCreator(attr_type, attr_table)

        return attr_obj.__dict__

    def _collect_operator_metadata(
        self,
        block: TosaBasicBlock,
        operator_io_fn: Callable[[int], str],
        length_fn: Callable[[], int],
    ) -> List[gb.MetadataItem]:
        """
        Collect metadata for operator inputs or outputs.

        Args:
            block: The TOSA basic block.
            accessor: Function to access tensors by index.
            length_accessor: Function to get the number of tensors.

        Returns:
            List of MetadataItem objects.
        """
        items: List[gb.MetadataItem] = []

        for idx in range(length_fn()):
            tensor_name = safe_decode(operator_io_fn(idx))
            tensor = find_tensor_in_block(block, tensor_name)
            if not tensor:
                continue

            attr_map = {
                "tensor_name": tensor_name,
                "shape": [tensor.Shape(i) for i in range(tensor.ShapeLength())]
                if tensor.ShapeLength() > 0
                else [],
                "type": tensor.Type(),
                "variable": tensor.Variable(),
                "is_unranked": tensor.IsUnranked(),
                "variable_name": tensor.VariableName(),
            }

            metadata_attrs = dict_to_key_value_list(attr_map)
            items.append(gb.MetadataItem(id=tensor_name, attrs=metadata_attrs))

        return items

