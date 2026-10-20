# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import glob
import json
import os
from dataclasses import asdict
from importlib.metadata import version

import pytest
from packaging.version import Version

from ..builder import DEFAULT_ELEMENT_COUNT, TosaGraphBuilder

FIXTURES_ROOT = os.path.join(os.path.dirname(__file__), "fixtures")

test_case_dirs = [
    d for d in glob.glob(os.path.join(FIXTURES_ROOT, "*")) if os.path.isdir(d)
]


@pytest.mark.parametrize(
    "case_dir", test_case_dirs, ids=lambda d: os.path.basename(d)
)
def test_e2e(case_dir):
    """Test parsing for each TOSA flatbuffer file and compare against expected graph output."""

    input_tosa = os.path.join(case_dir, "input.tosa")

    model_explorer_version = Version(version("ai-edge-model-explorer"))
    if model_explorer_version >= Version("0.1.33"):
        # Model Explorer v0.1.33 adds a "graphSorting" section to the JSON,
        # requiring a slightly different expected result for newer versions.
        expected_filename = "expected_with_graph_sorting.json"
    else:
        expected_filename = "expected.json"

    expected_json = os.path.join(case_dir, expected_filename)

    settings = {"const_element_count_limit": DEFAULT_ELEMENT_COUNT}
    graph_collection = TosaGraphBuilder(input_tosa, settings).graph_collection

    got = asdict(graph_collection)

    with open(expected_json) as f:
        expected = json.load(f)

    assert got == expected, (
        f"Test failed for {input_tosa}. Expected and actual output differ."
    )
