# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from model_explorer import graph_builder as gb

from .tosa_1_0 import (
    Attribute,
    AttributeCreator,
    Op,
    TosaBasicBlock,
    TosaGraph,
    TosaOperator,
    TosaShapeT,
    TosaTensorT,
)
from .util import (
    dict_to_key_value_list,
    enum_name,
    iter_vector,
    operator_id,
    read_file,
    safe_decode,
)

DEFAULT_ELEMENT_COUNT = 16
CONST_NODE_LABELS_TO_HIDE = ["CONST", "CONST_SHAPE"]


class TosaGraphBuilder:
    """
    A parser for TOSA FlatBuffer files.

    This class reads a TOSA FlatBuffer file, and converts contained graphs,
    blocks, operators, and tensors into a
    GraphCollection for visualization.
    """

    def __init__(self, file_path: str, settings: Dict):
        """
        Initialize the parser for a given FlatBuffer file.

        Args:
            file_path: Path to the TOSA FlatBuffer file to parse.
            settings: Settings dictionary from Model Explorer.
        """
        tosa_graph = TosaGraph.GetRootAsTosaGraph(read_file(file_path), 0)
        self._check_version(tosa_graph)

        self._input_node_id = "0"
        self._output_node_id = "-1"

        self._const_element_count_limit = (
            settings.get("const_element_count_limit", DEFAULT_ELEMENT_COUNT)
            if settings
            else DEFAULT_ELEMENT_COUNT
        )

        self._region_id_map: Dict[str, str] = {}
        self._decode_cache: Dict[Any, str] = {}
        self._metadata_cache: Dict[str, gb.MetadataItem] = {}
        self.graph_collection = self._build_graph_collection(
            file_path, tosa_graph
        )

    def _check_version(self, tosa_graph: TosaGraph):
        """
        Check if the TOSA version is supported.

        Raises:
            ValueError: If the TOSA version is less than 1.0.
        """
        version_obj = tosa_graph.Version()
        if version_obj is None:
            return
        if version_obj._Major() < 1:
            raise ValueError(
                f"Unsupported TOSA version: {version_obj._Major()}.{version_obj._Minor()}. Expected >= 1.0."
            )

    def _build_graph_collection(
        self, file_path: str, tosa_graph: TosaGraph
    ) -> gb.GraphCollection:
        """
        Parse the FlatBuffer into a GraphCollection object.

        Processes each region and its basic blocks, converting them into
        Graph objects and aggregating into a GraphCollection.

        Returns:
            A GraphCollection containing one or more parsed graphs.
        """
        self._identify_regions(tosa_graph)

        graphs: List[gb.Graph] = []

        for region in iter_vector(
            tosa_graph.Regions, tosa_graph.RegionsLength()
        ):
            region_name = self._decode(region.Name())
            for block in iter_vector(region.Blocks, region.BlocksLength()):
                graphs.append(self._build_graph(block, region_name))

        return gb.GraphCollection(label=Path(file_path).stem, graphs=graphs)

    def _identify_regions(self, tosa_graph: TosaGraph):
        """
        Identify all regions in the root graph and map their names to IDs.

        Populates self.region_id_map, assigning a default name "region{i}" if
        no explicit name is provided in the FlatBuffer.
        """
        for idx, region in enumerate(
            iter_vector(tosa_graph.Regions, tosa_graph.RegionsLength())
        ):
            region_name = self._decode(region.Name())
            if not region_name:
                region_name = f"region{idx}"
            self._region_id_map[region_name] = region_name

    def _decode(self, value: Any) -> Any:
        """Memoized decode of bytes to string."""
        if value not in self._decode_cache:
            self._decode_cache[value] = safe_decode(value)

        return self._decode_cache[value]

    def _build_graph(
        self,
        block: TosaBasicBlock,
        graph_id: str,
    ) -> gb.Graph:
        """
        Convert a TOSA basic block into a Graph object.

        Args:
            block: The TosaBasicBlock object to process.
            graph_id: Unique identifier (namespace) for this graph.

        Returns:
            A gb.Graph instance representing the block and its operators.
        """
        op_input_map: Dict[str, TosaShapeT | TosaTensorT] = {
            self._decode(item.Name()): TosaTensorT.InitFromObj(item)
            for item in iter_vector(block.Tensors, block.TensorsLength())
        } | {
            self._decode(item.Name()): TosaShapeT.InitFromObj(item)
            for item in iter_vector(block.Shapes, block.ShapesLength())
        }

        producer_map = self._map_outputs(block, graph_id)
        io_nodes = self._build_io_nodes(block, op_input_map, producer_map)
        op_nodes = self._build_operator_nodes(
            block, graph_id, op_input_map, producer_map
        )
        return gb.Graph(
            id=graph_id,
            nodes=io_nodes + op_nodes,
            nodeLabelsToHide=CONST_NODE_LABELS_TO_HIDE.copy(),
        )

    def _map_outputs(
        self, block: TosaBasicBlock, namespace: str
    ) -> Dict[str, str]:
        """
        Map each tensor name to the ID of the node that produces it.

        Args:
            block: The BasicBlock containing operator definitions.
            namespace: Namespace prefix used for operator node IDs.

        Returns:
            Dict[str, str] mapping tensor name to producer node ID.
        """
        output_map: Dict[str, str] = {}

        for input in iter_vector(block.Inputs, block.InputsLength()):
            output_map[self._decode(input)] = self._input_node_id

        for idx, op in enumerate(
            iter_vector(block.Operators, block.OperatorsLength())
        ):
            for output in iter_vector(op.Outputs, op.OutputsLength()):
                output_map[self._decode(output)] = operator_id(namespace, idx)

        return output_map

    def _build_io_nodes(
        self,
        block: TosaBasicBlock,
        op_input_map: Dict[str, TosaShapeT | TosaTensorT],
        producer_map: Dict[str, str],
    ) -> List[gb.GraphNode]:
        """
        Build graph I/O nodes (inputs and outputs) for a block.
        """
        nodes: List[gb.GraphNode] = []
        inp = self._build_input_node(block, op_input_map)
        if inp:
            nodes.append(inp)
        out = self._build_output_node(block, op_input_map, producer_map)
        if out:
            nodes.append(out)
        return nodes

    def _build_input_node(
        self,
        block: TosaBasicBlock,
        op_input_map: Dict[str, TosaShapeT | TosaTensorT],
    ) -> Optional[gb.GraphNode]:
        """
        Build the GraphInputs node for this block if inputs exist.

        Args:
            block: The BasicBlock object containing input tensor references.
            op_input_map: Lookup for tensor or const_shape metadata.

        Returns:
            A GraphNode labeled "GraphInputs" or None if no inputs.
        """
        if block.InputsLength():
            return gb.GraphNode(
                id=self._input_node_id,
                label="GraphInputs",
                namespace="GraphInputs",
                outputsMetadata=self._collect_metadata(
                    iter_vector(block.Inputs, block.InputsLength()),
                    op_input_map,
                ),
            )

    def _build_output_node(
        self,
        block: TosaBasicBlock,
        op_input_map: Dict[str, TosaShapeT | TosaTensorT],
        tensor_producer_map: Dict[str, str],
    ) -> Optional[gb.GraphNode]:
        """
        Build the GraphOutputs node for this block if outputs exist.

        Args:
            block: The BasicBlock object containing output tensor references.
            op_input_map: Lookup for tensor or const_shape metadata.
            tensor_producer_map: Mapping from tensor name to producer node ID.

        Returns:
            A GraphNode labeled "GraphOutputs" or None if no outputs.
        """
        if block.OutputsLength():
            return gb.GraphNode(
                id=self._output_node_id,
                label="GraphOutputs",
                namespace="GraphOutputs",
                inputsMetadata=self._collect_metadata(
                    iter_vector(block.Outputs, block.OutputsLength()),
                    op_input_map,
                ),
                incomingEdges=self._add_incoming_edges(
                    iter_vector(block.Outputs, block.OutputsLength()),
                    tensor_producer_map,
                    set_target=False,
                ),
            )

    def _build_operator_nodes(
        self,
        block: TosaBasicBlock,
        graph_id: str,
        op_input_map: Dict[str, TosaShapeT | TosaTensorT],
        producer_map: Dict[str, str],
    ) -> List[gb.GraphNode]:
        """
        Build GraphNode objects for all non-constant operators in the block.
        """
        nodes: List[gb.GraphNode] = []
        for idx, op in enumerate(
            iter_vector(block.Operators, block.OperatorsLength())
        ):
            op_attrs = self._collect_operator_attrs(op)
            node = gb.GraphNode(
                id=operator_id(graph_id, idx),
                label=enum_name(op.Op(), Op),
                namespace=graph_id,
                incomingEdges=self._add_incoming_edges(
                    iter_vector(op.Inputs, op.InputsLength()), producer_map
                ),
                attrs=dict_to_key_value_list(
                    op_attrs, self._const_element_count_limit
                ),
                inputsMetadata=self._collect_metadata(
                    iter_vector(op.Inputs, op.InputsLength()), op_input_map
                ),
                outputsMetadata=self._collect_metadata(
                    iter_vector(op.Outputs, op.OutputsLength()), op_input_map
                ),
                subgraphIds=self._extract_subgraph_ids(
                    op.AttributeType(), op_attrs
                ),
            )
            nodes.append(node)
        return nodes

    def _add_incoming_edges(
        self,
        tensors: Iterable[str],
        tensor_producer_map: Dict[str, str],
        set_target: bool = True,
    ) -> List[gb.IncomingEdge]:
        """
        Generate list of IncomingEdge linking operator inputs to producers.

        Args:
            operator: The TOSA operator object.
            tensors: Iterable of tensor names.
            tensor_producer_map: Map from tensor name to source node ID.

        Returns:
            List of gb.IncomingEdge objects for existing inputs.
        """
        incoming_edges: List[gb.IncomingEdge] = []

        for tensor in tensors:
            name = self._decode(tensor)
            source_node_id = tensor_producer_map.get(name)

            if source_node_id:
                incoming_edges.append(
                    gb.IncomingEdge(
                        sourceNodeId=source_node_id,
                        sourceNodeOutputId=name,
                        targetNodeInputId=name if set_target else "0",
                    )
                )

        return incoming_edges

    def _collect_metadata(
        self,
        io_list: Iterable[str],
        op_input_map: Dict[str, TosaShapeT | TosaTensorT],
    ) -> List[gb.MetadataItem]:
        """
        Collect metadata for a list of tensor names.

        Args:
            io_list: List of tensor names (bytes or FlatBuffer string).
            op_input_map: Mapping from tensor or const_shape name to tensor object.

        Returns:
            List of MetadataItem with id and attribute list for each tensor.
        """
        items: List[gb.MetadataItem] = []
        for io in io_list:
            name = self._decode(io)
            tensor = op_input_map.get(name)
            if tensor is None:
                continue
            metadata = self._metadata_cache.get(name)
            if metadata is None:
                metadata = self._build_metadata_item(name, tensor)
                self._metadata_cache[name] = metadata
            items.append(metadata)
        return items

    def _build_metadata_item(
        self, name: str, tensor: TosaShapeT | TosaTensorT
    ) -> gb.MetadataItem:
        """Build display metadata for one tensor or shape."""
        is_shape = isinstance(tensor, TosaShapeT)
        return gb.MetadataItem(
            id=name,
            attrs=dict_to_key_value_list(
                tensor.__dict__,
                self._const_element_count_limit,
                is_shape=is_shape,
            ),
        )

    def _collect_operator_attrs(
        self,
        op: TosaOperator,
    ) -> Dict[str, Any]:
        """
        Retrieve the raw attribute dictionary from a TOSA operator.

        Args:
            op: The TosaOperator object.

        Returns:
            Dict[str, Any] of attribute names and their values.
        """
        attribute_table = op.Attribute()
        if attribute_table is None:
            return {}

        attr_type = op.AttributeType()
        attributes = AttributeCreator(attr_type, attribute_table).__dict__
        loc = op.Location()
        if loc is not None:
            attributes["opLocation"] = self._decode(loc.Text())
        return attributes

    def _extract_subgraph_ids(
        self,
        attr_type: int,
        op_attrs: Dict[str, Any],
    ) -> List[str]:
        """Extract conditional subgraph IDs from a TOSA operator attribute."""
        attr_mappings = {
            Attribute.WhileLoopAttribute: ["condGraph", "bodyGraph"],
            Attribute.CondIfAttribute: ["thenGraph", "elseGraph"],
        }

        if attr_type not in attr_mappings:
            return []

        subgraph_ids = []
        for attr_name in attr_mappings[attr_type]:
            graph_name = self._decode(op_attrs.get(attr_name))
            if graph_name and graph_name in self._region_id_map:
                subgraph_ids.append(self._region_id_map[graph_name])

        return subgraph_ids
