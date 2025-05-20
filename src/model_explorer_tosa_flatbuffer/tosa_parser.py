from pathlib import Path
from typing import List, Dict, Any, Union

from .util import read_file, operator_id, dict_to_key_value_list, safe_decode, enum_name
import tosa_0_8
import tosa_1_0

from model_explorer import graph_builder as gb

TosaBasicBlockTType = Union[tosa_0_8.TosaBasicBlockT, tosa_1_0.TosaBasicBlockT]
TosaOperatorTType = Union[tosa_0_8.TosaOperatorT, tosa_1_0.TosaOperatorT]


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
        self.tosa_module = self._detect_version()

        self.TosaGraph = getattr(self.tosa_module, "TosaGraph")
        self.TosaGraphT = getattr(self.tosa_module, "TosaGraphT")

        graph_obj = self.TosaGraph.GetRootAsTosaGraph(self.buffer, 0)
        self.root_graph = self.TosaGraphT.InitFromObj(graph_obj)

        self.input_node_id = "0"
        self.output_node_id = "-1"

    def _detect_version(self):
        """
        Peek at the TOSA version before fully parsing the file.
        Returns:
            The appropriate TOSA module to use for parsing.
        """
        tosa_graph = tosa_1_0.TosaGraph.GetRootAsTosaGraph(self.buffer, 0)
        version_obj = tosa_graph.Version()
        if not version_obj:
            return tosa_1_0

        tosa_version = (version_obj._Major(), version_obj._Minor())

        tosa_module = tosa_0_8 if tosa_version[0] < 1 else tosa_1_0
        return tosa_module

    def parse(self) -> gb.GraphCollection:
        """
        Parse the TOSA file into a GraphCollection.

        Returns:
            GraphCollection containing the parsed graph.
        """
        graphs: List[gb.Graph] = []

        for region in self.root_graph.regions:
            for idx, block in enumerate(region.blocks):
                graph_id = safe_decode(block.name) or f"block{idx}"
                graphs.append(
                    gb.Graph(id=graph_id, nodes=self._build_nodes(block, graph_id))
                )

        return gb.GraphCollection(label=Path(self.file_path).stem, graphs=graphs)

    def _build_nodes(
        self,
        block: TosaBasicBlockTType,
        namespace: str,
    ) -> List[gb.GraphNode]:
        """
        Build nodes from the TOSA Basic Block.

        Args:
            block: The TOSA basic block.
            namespace: Namespace for the node IDs.
        """
        nodes: List[gb.GraphNode] = []
        tensor_producer_map = self._map_outputs(block, namespace)

        input_node = gb.GraphNode(
            id=self.input_node_id,
            label="GraphInputs",
            namespace="GraphInputs",
            outputsMetadata=self._collect_operator_metadata(block, block.inputs),
        )

        output_node = gb.GraphNode(
            id=self.output_node_id,
            label="GraphOutputs",
            namespace="GraphOutputs",
            inputsMetadata=self._collect_operator_metadata(block, block.outputs),
            incomingEdges=[
                gb.IncomingEdge(
                    sourceNodeId=tensor_producer_map.get(safe_decode(output)) or "",
                    sourceNodeOutputId=safe_decode(output),
                ) for output in block.outputs
            ],
        )
        nodes.extend([input_node, output_node])

        for idx, op in enumerate(block.operators):
            name = enum_name(op.op, self.tosa_module.Op)
            if name == "CONST":
                continue

            nodes.append(
                gb.GraphNode(
                    id=operator_id(namespace, idx),
                    label=name,
                    namespace=namespace,
                    incomingEdges=self._add_incoming_edges(op, tensor_producer_map),
                    attrs=dict_to_key_value_list(
                        self._collect_operator_attrs(op), self.tosa_module
                    ),
                    inputsMetadata=self._collect_operator_metadata(block, op.inputs),
                    outputsMetadata=self._collect_operator_metadata(block, op.outputs),
                )
            )

        return nodes

    def _map_outputs(
        self, block: TosaBasicBlockTType, namespace: str
    ) -> Dict[str, str]:
        """
        Map tensor names to the operators that produce them.

        Args:
            block: The TOSA basic block.
            namespace: Namespace for the node IDs.

        Returns:
            Dictionary mapping tensor names to producer node IDs.
        """
        output_map: Dict[str, str] = {}

        for input in block.inputs:
            output_map[safe_decode(input)] = self.input_node_id

        for idx, op in enumerate(block.operators):
            name = enum_name(op.op, self.tosa_module.Op)
            if name == "CONST":
                continue

            for output in op.outputs:
                output_map[safe_decode(output)] = operator_id(namespace, idx)

        return output_map

    def _add_incoming_edges(
        self, operator: TosaOperatorTType, tensor_producer_map: Dict[str, str]
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

        for input in operator.inputs:
            input_tensor_name = safe_decode(input)
            source_node_id = tensor_producer_map.get(input_tensor_name)

            if source_node_id:
                for output in operator.outputs:
                    incoming_edges.append(
                        gb.IncomingEdge(
                            sourceNodeId=source_node_id,
                            sourceNodeOutputId=safe_decode(output),
                            targetNodeInputId=input_tensor_name,
                        )
                    )

        return incoming_edges

    def _collect_operator_attrs(
        self,
        op: TosaOperatorTType,
    ) -> Dict[str, Any]:
        """
        Collect attributes from a tosa operator using the appropriate version module.

        Args:
        op: The TOSA operator containing the attribute

        Returns: Dictionary of parsed attribute fields
        """
        if not hasattr(op, "attribute") or op.attribute is None:
            return {}

        return op.attribute.__dict__

    def _collect_operator_metadata(
        self,
        block: TosaBasicBlockTType,
        io_list: List[str],
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

        for io in io_list:
            name = safe_decode(io)
            for tensor in block.tensors:
                if safe_decode(tensor.name) == name:
                    items.append(
                        gb.MetadataItem(
                            id=name,
                            attrs=dict_to_key_value_list(
                                tensor.__dict__, self.tosa_module
                            ),
                        )
                    )

        return items
