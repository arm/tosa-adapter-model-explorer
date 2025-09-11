# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.
from typing import Dict

from model_explorer import Adapter, AdapterMetadata, ModelExplorerGraphs

from .builder import TosaGraphBuilder


class TosaFlatbufferAdapter(Adapter):
    metadata = AdapterMetadata(
        id="tosa_flatbuffer_adapter",
        name="TOSA Flatbuffer Adapter",
        description="TOSA Flatbuffer adapter for Model Explorer",
        source_repo="https://github.com/arm/tosa-adapter-model-explorer",
        fileExts=["tosa"],
    )

    def convert(self, model_path: str, settings: Dict) -> ModelExplorerGraphs:
        return {"graphs": TosaGraphBuilder(model_path).graph_collection.graphs}
