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
        graph = graph_builder.Graph(id="tosa_flatbuffer")

        return {"graphs": [graph]}
