from typing import Dict
from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs
from .tosa_parser import TosaParser


class TosaFlatbufferAdapter(Adapter):
    metadata = AdapterMetadata(
        id="tosa_flatbuffer_adapter",
        name="TOSA Flatbuffer Adapter",
        description="",
        source_repo="https://github.com/Arm-Debug/model-explorer-tosa-flatbuffer",
        fileExts=["tosa"],
    )

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        parser = TosaParser(model_path)
        graph_collection = parser.parse()

        return {"graphs": graph_collection.graphs}
