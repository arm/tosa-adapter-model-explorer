from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs, graph_builder
from tosa.TosaGraph import TosaGraph
from tosa.Op import Op


class TosaFlatbufferAdapter(Adapter):

    metadata = AdapterMetadata(
        id="tosa_flatbuffer_adapter",
        name="TOSA Flatbuffer Adapter",
        description="",
        source_repo="https://github.com/Arm-Debug/model-explorer-tosa-flatbuffer",
        fileExts=["tosa"],
    )

    def __init__(self):
        super().__init__()

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        # with open(model_path, mode="rb") as file:
        #     buf = file.read()

        #     tosa_graph = TosaGraph.GetRootAs(buf, 0)

        #     for i in range(0, tosa_graph.RegionsLength()):
        #         region = tosa_graph.Regions(i)

        #         args = {}

        #         for j in range(0, region.BlocksLength()):
        #             block = region.Blocks(j)

        #             for k in range(0, block.TensorsLength()):
        #                 tensor = block.Tensors(k)
        #                 key = tensor.Name().decode("utf-8")
        #                 args[key] = {"name": key}

        #             for k in range(0, block.OperatorsLength()):
        #                 operator = block.Operators(k)
        #                 name = f"{self.op_name(operator.Op())}_"

        #                 for m in range(0, operator.InputsLength()):
        #                     input = operator.Inputs(m).decode("utf-8")
        #                     incomingEdges.append(
        #                         {
        #                             "source_node_id": input,
        #                             "source_node_output_id": "output",
        #                             "target_node_input_id": f"input{m}",
        #                         }
        #                     )

        #                 nodes[] = {
        #                     "id": f"{region.Name().decode('utf-8')}_op_{k}",
        #                     "label": self.op_name(operator.Op()),
        #                     "namespace": "",
        #                     "incoming_edges": incomingEdges,
        #                 }

        #                 for m in range(0, operator.OutputsLength()):
        #                     output = operator.Outputs(m).decode("utf-8")
        #                     if nodes[output]:
        #                         if "incoming_edges" not in nodes[output]:
        #                             nodes[output]["incoming_edges"] = []

        #                         nodes[output]["incoming_edges"].append(
        #                             {
        #                                 "source_node_id": f"{region.Name().decode('utf-8')}_op_{k}",
        #                                 "source_node_output_id": "output",
        #                                 "target_node_input_id": f"input{m}",
        #                             }
        #                         )

        # print(nodes)

        graph = graph_builder.Graph(id="tosa_flatbuffer")

        # for key in nodes:
        #     graph_node = graph_builder.GraphNode(
        #         id=nodes[key]["id"],
        #         label=nodes[key]["label"],
        #         namespace=nodes[key]["namespace"],
        #     )

        #     if "incoming_edges" in nodes[key]:
        #         for incoming_edge in nodes[key]["incoming_edges"]:
        #             graph_builder.IncomingEdge(
        #                 sourceNodeId=incoming_edge["source_node_id"],
        #                 sourceNodeOutputId=incoming_edge["source_node_output_id"],
        #                 targetNodeInputId=incoming_edge["target_node_input_id"],
        #             )

            # graph.nodes.extend([graph_node])

        return {"graphs": [graph]}

    def convert_tosa_graph(self, tosa_graph):
        pass

    def op_name(self, op_value):
        for name in dir(Op):
            if not name.startswith("_") and getattr(Op, name) == op_value:
                return name
        return f"UNKNOWN({op_value})"
