# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License v2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for license information.

import struct

import pytest

from ..util import _decode_data, _DecodedDataPreview, dict_to_key_value_list


@pytest.mark.parametrize(
    ("dtype", "raw", "expected"),
    [
        ("BOOL", b"\x00\x01", [False, True]),
        ("INT8", struct.pack("<bb", -1, 2), [-1, 2]),
        ("INT16", struct.pack("<hh", -2, 3), [-2, 3]),
        ("INT32", struct.pack("<ii", -3, 4), [-3, 4]),
        (
            "INT48",
            (-4).to_bytes(6, "little", signed=True)
            + (5).to_bytes(6, "little", signed=True),
            [-4, 5],
        ),
        ("FP16", struct.pack("<ee", 1.5, -2.0), [1.5, -2.0]),
        ("FP32", struct.pack("<ff", 1.25, -2.5), [1.25, -2.5]),
        ("SHAPE", struct.pack("<qq", 1, 2), [1, 2]),
    ],
)
def test_decode_data_supported_dtypes(dtype, raw, expected):
    assert _decode_data(raw, dtype) == pytest.approx(expected)


@pytest.mark.parametrize(
    "dtype", ["UNKNOWN", "INT4", "BF16", "FP8E4M3", "FP8E5M2"]
)
def test_decode_data_raw_fallback_dtypes(dtype):
    raw = b"\x01\x02\x03"

    assert _decode_data(raw, dtype) == raw


def test_decode_data_accepts_list_backed_flatbuffer_data():
    attrs = dict_to_key_value_list({"type": 5, "data": [1, 0, 0, 0]}, 16)

    assert attrs[1].value == "[1]"


def test_decode_data_can_limit_preview_without_losing_total_count():
    raw = struct.pack("<10i", *range(10))
    decoded = _decode_data(raw, "INT32", max_elements=4)

    assert isinstance(decoded, _DecodedDataPreview)
    assert len(decoded) == 10
    assert list(decoded) == [0, 1, 2, 3]


def test_dict_to_key_value_list_decodes_only_display_preview():
    raw = struct.pack("<10i", *range(10))
    attrs = dict_to_key_value_list({"type": 5, "data": raw}, 3)

    assert attrs[1].value == ("(showing 3 out of 10 elements)\n[0, 1, 2...]")


@pytest.mark.parametrize(
    ("dtype", "raw", "expected"),
    [
        ("INT16", struct.pack("<h", 1) + b"\xff", [1]),
        ("INT32", struct.pack("<i", 2) + b"\xff", [2]),
        (
            "INT48",
            (3).to_bytes(6, "little", signed=True) + b"\xff",
            [3],
        ),
        ("FP16", struct.pack("<e", 1.5) + b"\xff", [1.5]),
        ("FP32", struct.pack("<f", 2.5) + b"\xff", [2.5]),
        ("SHAPE", struct.pack("<q", 4) + b"\xff", [4]),
    ],
)
def test_decode_data_ignores_trailing_malformed_bytes(dtype, raw, expected):
    assert _decode_data(raw, dtype) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("dtype", "raw"),
    [
        ("INT16", b"\x01"),
        ("INT32", b"\x01\x02\x03"),
        ("INT48", b"\x01\x02\x03\x04\x05"),
        ("FP16", b"\x01"),
        ("FP32", b"\x01\x02\x03"),
        ("SHAPE", b"\x01\x02\x03\x04\x05\x06\x07"),
    ],
)
def test_decode_data_returns_empty_for_incomplete_element(dtype, raw):
    assert _decode_data(raw, dtype) == []
