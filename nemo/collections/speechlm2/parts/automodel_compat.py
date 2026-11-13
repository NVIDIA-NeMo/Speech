# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import logging


logger = logging.getLogger(__name__)

_HF_CONFIG_KWARGS = ("cache_dir", "revision", "token", "local_files_only", "subfolder")


_MTP_KWARGS = ("mtp_loss_scaling_factor", "mtp_use_repeated_layer")


def remove_automodel_kwargs_for_hf_fallback(
    model_path_or_name: str,
    kwargs: dict,
    *,
    trust_remote_code: bool = False,
    mtp_config_overrides: dict | None = None,
) -> bool:
    """Remove Automodel-only kwargs before its Hugging Face fallback.

    Automodel consumes ``backend`` and the MTP options for native model
    implementations but forwards them to Hugging Face model constructors on the
    fallback path. Those constructors do not accept these Automodel-specific
    keywords. Resolve the same native-vs-HF choice up front and remove the
    incompatible kwargs that are no-ops for an HF model: ``backend`` and a
    disabled MTP depth (``num_nextn_predict_layers=0``). Requesting MTP for an
    HF fallback model raises, since the HF implementation cannot build an MTP head.

    Returns:
        ``True`` when any kwarg was removed; otherwise ``False``.

    Raises:
        ValueError: When MTP is requested for a model without a native Automodel implementation.
    """
    disables_mtp = "num_nextn_predict_layers" in kwargs and not kwargs["num_nextn_predict_layers"]
    requests_mtp = (
        mtp_config_overrides is not None
        or any(key in kwargs for key in _MTP_KWARGS)
        or ("num_nextn_predict_layers" in kwargs and not disables_mtp)
    )
    if "backend" not in kwargs and not disables_mtp and not requests_mtp:
        return False

    try:
        from nemo_automodel._transformers.model_init import get_is_hf_model
        from transformers import AutoConfig

        config = kwargs.get("config")
        if config is None:
            config_kwargs = {key: kwargs[key] for key in _HF_CONFIG_KWARGS if key in kwargs}
            config = AutoConfig.from_pretrained(
                model_path_or_name,
                trust_remote_code=trust_remote_code,
                **config_kwargs,
            )

        uses_hf_model = kwargs.get("quantization_config") is not None or get_is_hf_model(
            config,
            force_hf=bool(kwargs.get("force_hf", False)),
        )
    except Exception as error:
        logger.warning("Could not determine Automodel implementation; leaving kwargs unchanged: %s", error)
        return False

    if not uses_hf_model:
        return False

    if requests_mtp:
        raise ValueError(
            f"Model {model_path_or_name} uses the Hugging Face fallback implementation in NeMo Automodel, which "
            "does not support Multi-Token Prediction (MTP). Disable MTP (mtp.enabled=false, "
            "retain_native_mtp=false) or use an LLM with a native Automodel implementation."
        )

    removed = False
    if disables_mtp:
        # HF fallback models never build an MTP head, so a zero depth is already satisfied.
        kwargs.pop("num_nextn_predict_layers")
        removed = True
    if "backend" in kwargs:
        # Automodel still forwards backend to HF constructors on this fallback path.
        kwargs.pop("backend")
        logger.warning(
            "Ignoring Automodel backend configuration for Hugging Face fallback model %s", model_path_or_name
        )
        removed = True
    return removed
