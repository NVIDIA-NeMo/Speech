#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Relocate a sealed ShareGPT tar route to byte-identical local sources.

The route payload addresses tar shards by ordinal and members by local index,
so it remains valid when an immutable source set is copied to new paths. This
tool preserves that payload while recomputing every path- or filesystem-bound
header digest. It fails closed unless manifest bytes and collection shape
match the sealed source route.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import click
from scripts.dataloading._sharegpt_route_config import ShareGptRouteSpec, discover_sharegpt_route_specs
from scripts.dataloading.build_indexes import _load_input_cfg

from nemo.collections.common.data.lhotse.sharegpt_tar_routing import (
    ShareGptTarRoutingIndex,
    canonical_audio_prefix_map_digest,
    ordered_manifest_content_digest,
    ordered_manifest_content_digest_from_mirrors,
    ordered_manifest_source_identity_digest,
    ordered_manifest_source_identity_digest_from_metadata,
    ordered_manifest_spec_path_digest,
    ordered_tar_catalog_digest,
    ordered_tar_catalog_digest_from_metadata,
    write_sharegpt_tar_routing_index,
)


def _sha256_file(path: str | Path) -> bytes:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def relocate_sharegpt_route(
    source_route: str | Path,
    output_route: str | Path,
    *,
    spec: ShareGptRouteSpec,
    tar_metadata: Sequence[Mapping] | None = None,
    manifest_mirrors: Sequence[str | Path] | None = None,
    manifest_metadata: Sequence[Mapping] | None = None,
    manifest_content_digest: bytes | str | None = None,
    source_tar_paths: Sequence[str | Path] | None = None,
    trust_relocated_tar_payloads: bool = False,
) -> Path:
    """Publish a route for ``spec`` while preserving a sealed route payload."""
    source_route = Path(source_route)
    output_route = Path(output_route)
    if source_route.resolve() == output_route.resolve():
        raise ValueError("Source and output route paths must differ")

    with ShareGptTarRoutingIndex(source_route) as source:
        header = source.header
        if len(spec.tar_paths) != header.tar_shard_count:
            raise ValueError(
                "Tar collection size changed during route relocation: "
                f"{len(spec.tar_paths)} != {header.tar_shard_count}"
            )

        offline_manifest_sources = sum(value is not None for value in (manifest_mirrors, manifest_content_digest))
        if manifest_metadata is None and offline_manifest_sources:
            raise ValueError("Offline manifest relocation requires metadata")
        if manifest_metadata is not None and offline_manifest_sources != 1:
            raise ValueError("Offline manifest relocation requires exactly one of mirrors " "or a content digest")
        if manifest_metadata is None:
            content_digest = ordered_manifest_content_digest(spec.manifest_paths)
            manifest_identity_digest = ordered_manifest_source_identity_digest(spec.manifest_paths)
        elif manifest_content_digest is not None:
            content_digest = _coerce_sha256_digest(manifest_content_digest, "manifest content")
            manifest_identity_digest = ordered_manifest_source_identity_digest_from_metadata(
                spec.manifest_paths, manifest_metadata
            )
        else:
            content_digest = ordered_manifest_content_digest_from_mirrors(spec.manifest_paths, manifest_mirrors)
            manifest_identity_digest = ordered_manifest_source_identity_digest_from_metadata(
                spec.manifest_paths, manifest_metadata
            )
        if not hmac.compare_digest(content_digest, header.manifest_content_digest):
            raise ValueError(
                "Manifest content changed during route relocation; rebuilding the "
                "route from source indexes is required"
            )

        tar_catalog_digest = (
            ordered_tar_catalog_digest(spec.tar_paths)
            if tar_metadata is None
            else ordered_tar_catalog_digest_from_metadata(spec.tar_paths, tar_metadata)
        )
        if not hmac.compare_digest(tar_catalog_digest, header.tar_catalog_digest):
            if trust_relocated_tar_payloads:
                pass
            elif source_tar_paths is None:
                raise ValueError(
                    "Tar paths or identities changed, but target payload equivalence "
                    "was not proven. Pass the ordered original paths with source_tar_paths "
                    "or explicitly trust an external attestation."
                )
            else:
                if len(source_tar_paths) != len(spec.tar_paths):
                    raise ValueError(
                        "Expected one source tar path per target tar path, got "
                        f"{len(source_tar_paths)} and {len(spec.tar_paths)}"
                    )
                source_catalog_digest = ordered_tar_catalog_digest(source_tar_paths)
                if not hmac.compare_digest(source_catalog_digest, header.tar_catalog_digest):
                    raise ValueError("Provided source tar paths do not match the sealed route catalog.")
                for index, (source_path, target_path) in enumerate(zip(source_tar_paths, spec.tar_paths, strict=True)):
                    if "://" in str(source_path) or "://" in str(target_path):
                        raise ValueError(
                            "Cannot hash remote tar relocation at position "
                            f"{index}; use an externally verified transfer attestation."
                        )
                    if not hmac.compare_digest(_sha256_file(source_path), _sha256_file(target_path)):
                        raise ValueError(
                            "Relocated tar payload content mismatch at position "
                            f"{index}: {source_path!r} != {target_path!r}"
                        )

        rows = (source.routes_for_row(index) for index in range(len(source)))
        return write_sharegpt_tar_routing_index(
            output_route,
            rows,
            tar_shard_count=len(spec.tar_paths),
            manifest_spec_path_digest=ordered_manifest_spec_path_digest(spec.manifest_paths, spec.manifest_specs),
            manifest_content_digest=content_digest,
            manifest_source_identity_digest=manifest_identity_digest,
            tar_catalog_digest=tar_catalog_digest,
            audio_prefix_map_digest=canonical_audio_prefix_map_digest(spec.audio_prefix_map),
        )


def select_route_spec(
    input_cfg: str | Path,
    *,
    data_blend_dir: str | Path | None,
) -> ShareGptRouteSpec:
    """Load exactly one collection-mode route specification from a config."""
    config = _load_input_cfg(str(input_cfg), data_blend_dir)
    specs = discover_sharegpt_route_specs(config, data_blend_dir=data_blend_dir)
    if len(specs) != 1:
        raise ValueError(f"Expected exactly one collection-mode route, discovered {len(specs)}")
    return specs[0]


@click.command()
@click.argument("input_cfg", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--source-route",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option("--data-blend-dir", type=click.Path(file_okay=False))
@click.option(
    "--tar-metadata",
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Sealed JSON object identities for offline remote relocation. The file "
        "must contain schema_version 1 and one ordered record per tar path."
    ),
)
@click.option(
    "--manifest-mirror",
    "manifest_mirrors",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Local content mirror for one ordered manifest; repeat in config order.",
)
@click.option(
    "--manifest-metadata",
    type=click.Path(exists=True, dir_okay=False),
    help="Sealed JSON local identities matching the ordered manifest mirrors.",
)
@click.option(
    "--manifest-content-digest",
    help=(
        "Precomputed 64-hex ordered manifest-content digest. Requires "
        "--manifest-metadata and cannot be combined with --manifest-mirror."
    ),
)
@click.option(
    "--source-tar",
    "source_tar_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Ordered original tar path used to prove byte identity after a local "
        "relocation; repeat once per target tar in config order."
    ),
)
@click.option(
    "--trust-relocated-tar-payloads",
    is_flag=True,
    help=(
        "Accept an externally attested tar relocation without hashing source "
        "and target payloads. Required for relocated remote objects."
    ),
)
def main(
    input_cfg: str,
    source_route: str,
    output: str,
    data_blend_dir: str | None,
    tar_metadata: str | None,
    manifest_mirrors: tuple[str, ...],
    manifest_metadata: str | None,
    manifest_content_digest: str | None,
    source_tar_paths: tuple[str, ...],
    trust_relocated_tar_payloads: bool,
) -> None:
    """Relocate SOURCE_ROUTE to the paths declared by INPUT_CFG."""
    try:
        spec = select_route_spec(input_cfg, data_blend_dir=data_blend_dir)
        metadata = None
        if tar_metadata is not None:
            metadata = _load_metadata_catalog(tar_metadata, "Tar")
        manifest_records = None
        if manifest_metadata is not None:
            manifest_records = _load_metadata_catalog(manifest_metadata, "Manifest")
        offline_sources = int(bool(manifest_mirrors)) + int(manifest_content_digest is not None)
        if manifest_records is None and offline_sources:
            raise ValueError("Offline manifest relocation requires --manifest-metadata")
        if manifest_records is not None and offline_sources != 1:
            raise ValueError(
                "Offline manifest relocation requires exactly one of " "--manifest-mirror or --manifest-content-digest"
            )
        if source_tar_paths and trust_relocated_tar_payloads:
            raise ValueError("--source-tar and --trust-relocated-tar-payloads are mutually exclusive")
        result = relocate_sharegpt_route(
            source_route,
            output,
            spec=spec,
            tar_metadata=metadata,
            manifest_mirrors=manifest_mirrors or None,
            manifest_metadata=manifest_records,
            manifest_content_digest=manifest_content_digest,
            source_tar_paths=source_tar_paths or None,
            trust_relocated_tar_payloads=trust_relocated_tar_payloads,
        )
    except (OSError, TypeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Wrote sealed route: {result}")


def _load_metadata_catalog(path: str | Path, label: str) -> list[Mapping]:
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "objects",
    }:
        raise ValueError(f"{label} metadata must contain exactly schema_version and objects")
    if document["schema_version"] != 1 or not isinstance(document["objects"], list):
        raise ValueError(f"{label} metadata requires schema_version 1 and an objects list")
    return document["objects"]


def _coerce_sha256_digest(value: bytes | str, label: str) -> bytes:
    if isinstance(value, bytes):
        digest = value
    elif isinstance(value, str):
        try:
            digest = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError(f"Invalid {label} digest hex") from error
    else:
        raise TypeError(f"{label} digest must be bytes or hex text")
    if len(digest) != 32:
        raise ValueError(f"{label} digest must be exactly 32 bytes")
    return digest


if __name__ == "__main__":
    main()
