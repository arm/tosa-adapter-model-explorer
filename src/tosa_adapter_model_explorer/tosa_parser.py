# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from .util import read_file, operator_id, dict_to_key_value_list, safe_decode, enum_name
import tosa_0_8
import tosa_1_0

from model_explorer import graph_builder as gb

TosaBasicBlockTType = Union[tosa_0_8.TosaBasicBlockT, tosa_1_0.TosaBasicBlockT]
TosaOperatorTType = Union[tosa_0_8.TosaOperatorT, tosa_1_0.TosaOperatorT]


class TosaParser:
    """
    A parser for TOSA FlatBuffer files handling multiple schema versions (0.8 and 1.0).

    This class reads a TOSA FlatBuffer file, determines its schema version,
    and converts contained graphs, blocks, operators, and tensors into a
    GraphCollection for visualization.
    """

    def __init__(self, file_path: str):
        """
        Initialize the parser for a given FlatBuffer file.

        Args:
            file_path: Path to the TOSA FlatBuffer file to parse.
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

        self.region_id_map: Dict[str, str] = {}

    def _detect_version(self):
        """
        Determine the TOSA schema version for this model buffer.

        Reads the version field and selects the corresponding tosa_0_8 or
        tosa_1_0 Python module for parsing.

        Returns:
            The tosa_x_x module matching the detected schema version.
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
        Parse the FlatBuffer into a GraphCollection object.

        Processes each region and its basic blocks, converting them into
        Graph objects and aggregating into a GraphCollection.

        Returns:
            A GraphCollection containing one or more parsed graphs.
        """
        self._identify_regions()

        graphs: List[gb.Graph] = []

        for region in self.root_graph.regions:
            for block in region.blocks:
                region_name = safe_decode(region.name)
                graphs.append(self._build_graph(block, region_name))

        return gb.GraphCollection(label=Path(self.file_path).stem, graphs=graphs)

    def _identify_regions(self):
        """
        Identify all regions in the root graph and map their names to IDs.

        Populates self.region_id_map, assigning a default name "region{i}" if
        no explicit name is provided in the FlatBuffer.
        """
        for idx, region in enumerate(self.root_graph.regions):
            region_name = safe_decode(region.name)
            if not region_name:
                region_name = f"region{idx}"
            self.region_id_map[region_name] = region_name

    def _collect_metadata(
        self,
        io_list: List[Any],
        op_input_map: Dict[str, Any],
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
            name = safe_decode(io)
            tensor = op_input_map.get(name)
            if tensor is None:
                continue
            items.append(
                gb.MetadataItem(
                    id=name,
                    attrs=dict_to_key_value_list(tensor.__dict__, self.tosa_module),
                )
            )
        return items

    def _build_input_node(
        self,
        block: TosaBasicBlockTType,
        op_input_map: Dict[str, Any],
    ) -> Optional[gb.GraphNode]:
        """
        Build the GraphInputs node for this block if inputs exist.

        Args:
            block: The BasicBlock object containing input tensor references.
            op_input_map: Lookup for tensor or const_shape metadata.

        Returns:
            A GraphNode labeled "GraphInputs" or None if no inputs.
        """
        if block.inputs:
            return gb.GraphNode(
                id=self.input_node_id,
                label="GraphInputs",
                namespace="GraphInputs",
                outputsMetadata=self._collect_metadata(block.inputs, op_input_map),
            )

    def _build_output_node(
        self,
        block: TosaBasicBlockTType,
        op_input_map: Dict[str, Any],
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
        if block.outputs:
            return gb.GraphNode(
                id=self.output_node_id,
                label="GraphOutputs",
                namespace="GraphOutputs",
                inputsMetadata=self._collect_metadata(block.outputs, op_input_map),
                incomingEdges=[
                    gb.IncomingEdge(
                        sourceNodeId=tensor_producer_map.get(name, ""),
                        sourceNodeOutputId=name,
                    )
                    for name in (safe_decode(o) for o in block.outputs)
                ],
            )

    def _build_graph(
        self,
        block: TosaBasicBlockTType,
        graph_id: str,
    ) -> gb.Graph:
        """
        Convert a TOSA basic block into a Graph object.

        Args:
            block: The TosaBasicBlockT object to process.
            graph_id: Unique identifier (namespace) for this graph.

        Returns:
            A gb.Graph instance representing the block and its operators.
        """
        op_input_map = {safe_decode(item.name): item for item in block.tensors}
        shapes = getattr(block, "shapes", None) or []
        op_input_map.update({safe_decode(item.name): item for item in shapes})

        producer_map = self._map_outputs(block, graph_id)
        io_nodes = self._build_io_nodes(block, op_input_map, producer_map)
        op_nodes = self._build_operator_nodes(block, graph_id, op_input_map, producer_map)
        return gb.Graph(id=graph_id, nodes=io_nodes + op_nodes)

    def _map_outputs(
        self, block: TosaBasicBlockTType, namespace: str
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

        for input in block.inputs:
            output_map[safe_decode(input)] = self.input_node_id

        for idx, op in enumerate(block.operators):
            for output in op.outputs:
                output_map[safe_decode(output)] = operator_id(namespace, idx)

        return output_map

    def _add_incoming_edges(
        self, operator: TosaOperatorTType, tensor_producer_map: Dict[str, str]
    ) -> List[gb.IncomingEdge]:
        """
        Generate list of IncomingEdge linking operator inputs to producers.

        Args:
            operator: The TOSA operator object.
            tensor_producer_map: Map from tensor name to source node ID.

        Returns:
            List of gb.IncomingEdge objects for existing inputs.
        """
        incoming_edges: List[gb.IncomingEdge] = []

        for input in operator.inputs:
            input_tensor_name = safe_decode(input)
            source_node_id = tensor_producer_map.get(input_tensor_name)

            if source_node_id:
                incoming_edges.append(
                    gb.IncomingEdge(
                        sourceNodeId=source_node_id,
                        sourceNodeOutputId=input_tensor_name,
                        targetNodeInputId=input_tensor_name,
                    )
                )

        return incoming_edges

    def _extract_subgraph_ids(self, op: TosaOperatorTType) -> List[str]:
        """
        Extract conditional subgraph IDs from a TOSA operator attribute.
        """
        subgraph_ids: List[str] = []
        attr_type = enum_name(op.attributeType, self.tosa_module.Attribute)
        if attr_type == "CondIfAttribute" and op.attribute:
            tg = getattr(op.attribute, "thenGraph", None) or getattr(
                op.attribute, "thenBranch", None
            )
            eg = getattr(op.attribute, "elseGraph", None) or getattr(
                op.attribute, "elseBranch", None
            )
            then_name = safe_decode(tg) if tg is not None else None
            else_name = safe_decode(eg) if eg is not None else None
            if then_name and then_name in self.region_id_map:
                subgraph_ids.append(self.region_id_map[then_name])
            if else_name and else_name in self.region_id_map:
                subgraph_ids.append(self.region_id_map[else_name])
        return subgraph_ids

    def _build_io_nodes(
        self,
        block: TosaBasicBlockTType,
        op_input_map: Dict[str, Any],
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

    def _build_operator_nodes(
        self,
        block: TosaBasicBlockTType,
        graph_id: str,
        op_input_map: Dict[str, Any],
        producer_map: Dict[str, str],
    ) -> List[gb.GraphNode]:
        """
        Build GraphNode objects for all non-constant operators in the block.
        """
        nodes: List[gb.GraphNode] = []
        for idx, op in enumerate(block.operators):
            node = gb.GraphNode(
                id=operator_id(graph_id, idx),
                label=enum_name(op.op, self.tosa_module.Op),
                namespace=graph_id,
                incomingEdges=self._add_incoming_edges(op, producer_map),
                attrs=dict_to_key_value_list(
                    self._collect_operator_attrs(op), self.tosa_module
                ),
                inputsMetadata=self._collect_metadata(op.inputs, op_input_map),
                outputsMetadata=self._collect_metadata(op.outputs, op_input_map),
                subgraphIds=self._extract_subgraph_ids(op),
            )
            nodes.append(node)
        return nodes

    def _collect_operator_attrs(
        self,
        op: TosaOperatorTType,
    ) -> Dict[str, Any]:
        """
        Retrieve the raw attribute dictionary from a TOSA operator.

        Args:
            op: The TosaOperatorT object.

        Returns:
            Dict[str, Any] of attribute names and their values.
        """
        if not hasattr(op, "attribute") or op.attribute is None:
            return {}

        attributes = op.attribute.__dict__
        loc = getattr(op, "location", None)
        if self.tosa_module is tosa_1_0 and loc is not None:
            attributes['opLocation'] = safe_decode(loc.text)
        return attributes
