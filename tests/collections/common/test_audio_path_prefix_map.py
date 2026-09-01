# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from nemo.collections.common.data.lhotse.audio_path_resolver import AudioPathPrefixMap


def test_audio_path_prefix_map_leaves_relative_and_url_paths_unchanged():
    mapper = AudioPathPrefixMap({"/source/site-a": "/mirror/site-b"})

    assert mapper.resolve("audio/example.wav") == "audio/example.wav"
    assert (
        mapper.resolve("s3://existing-bucket/audio/example.wav")
        == "s3://existing-bucket/audio/example.wav"
    )


def test_audio_path_prefix_map_maps_local_and_remote_destinations():
    local = AudioPathPrefixMap({"/source/site-a": "/mirror/site-b"})
    remote = AudioPathPrefixMap(
        {"/source/site-a": "s3://example-audio/payload/site-a-source"}
    )

    assert (
        local.resolve("/source/site-a/audio/example.wav")
        == "/mirror/site-b/audio/example.wav"
    )
    assert (
        remote.resolve("/source/site-a/audio/example.wav")
        == "s3://example-audio/payload/site-a-source/audio/example.wav"
    )


def test_audio_path_prefix_map_uses_component_safe_longest_prefix():
    mapper = AudioPathPrefixMap(
        {
            "/source/site-a": "/mirror/general",
            "/source/site-a/special": "/mirror/special",
            "/source/site-a-other": "/mirror/other",
        }
    )

    assert mapper.resolve("/source/site-a/special/a.wav") == "/mirror/special/a.wav"
    assert (
        mapper.resolve("/source/site-a/ordinary/a.wav")
        == "/mirror/general/ordinary/a.wav"
    )
    assert mapper.resolve("/source/site-a-other/a.wav") == "/mirror/other/a.wav"


def test_audio_path_prefix_map_requires_all_absolute_paths_to_match_when_configured():
    mapper = AudioPathPrefixMap({"/source/site-a": "/mirror/site-b"})

    with pytest.raises(ValueError, match="does not match any configured source prefix"):
        mapper.resolve("/another/root/example.wav")


@pytest.mark.parametrize(
    "mapping,match",
    [
        (
            {"/source/site-a": "/mirror/a", "/source/site-a/": "/mirror/b"},
            "normalize to the same source prefix",
        ),
        ({"source/site-a": "/mirror/site-b"}, "must be an absolute POSIX path"),
        (
            {"/source/site-a/../escape": "/mirror/site-b"},
            "must not contain '.' or '..'",
        ),
        (
            {"/source/site-a": "relative/destination"},
            "must be an absolute POSIX path or URL",
        ),
        ({"/source/site-a": "s3://bucket/root?query=yes"}, "query or fragment"),
    ],
)
def test_audio_path_prefix_map_rejects_invalid_configuration(mapping, match):
    with pytest.raises(ValueError, match=match):
        AudioPathPrefixMap(mapping)


@pytest.mark.parametrize(
    "path", ["/source/site-a/../escape.wav", "/source/site-a/./audio.wav"]
)
def test_audio_path_prefix_map_rejects_input_traversal(path):
    mapper = AudioPathPrefixMap({"/source/site-a": "/mirror/site-b"})

    with pytest.raises(ValueError, match="must not contain '.' or '..'"):
        mapper.resolve(path)


def test_audio_path_prefix_map_digest_is_order_invariant_and_sensitive_to_destinations():
    first = AudioPathPrefixMap({"/a": "/x", "/b": "s3://bucket/y"})
    reordered = AudioPathPrefixMap({"/b": "s3://bucket/y", "/a": "/x"})
    changed = AudioPathPrefixMap({"/a": "/x", "/b": "s3://bucket/z"})

    assert first.digest == reordered.digest
    assert first.digest != changed.digest
    assert len(first.digest) == 64


def test_empty_audio_path_prefix_map_is_compatible_with_existing_absolute_paths():
    mapper = AudioPathPrefixMap({})

    assert mapper.resolve("/existing/local/audio.wav") == "/existing/local/audio.wav"
