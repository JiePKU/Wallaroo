# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for the inference libraries."""

import os
from glob import glob
from typing import Any

# import mediapy as media
import numpy as np
import torch

_DTYPE, _DEVICE = torch.bfloat16, "cuda"
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
_SPATIAL_ALIGN = 16
_TEMPORAL_ALIGN = 8


def load_model(
    jit_filepath: str = None,
    tokenizer_config: dict[str, Any] = None,
    device: str = "cuda",
) -> torch.nn.Module | torch.jit.ScriptModule:
    """Loads a torch.nn.Module from a filepath.

    Args:
        jit_filepath: The filepath to the JIT-compiled model.
        device: The device to load the model onto, default=cuda.
    Returns:
        The JIT compiled model loaded to device and on eval mode.
    """
    if tokenizer_config is None:
        return load_jit_model(jit_filepath, device)
    # full_model, ckpts = _load_pytorch_model(jit_filepath, tokenizer_config, device)
    # full_model.load_state_dict(ckpts.state_dict(), strict=True)
    # return full_model.eval().to(device)


def load_encoder_model(
    jit_filepath: str = None,
    tokenizer_config: dict[str, Any] = None,
    device: str = "cuda",
) -> torch.nn.Module | torch.jit.ScriptModule:
    """Loads a torch.nn.Module from a filepath.

    Args:
        jit_filepath: The filepath to the JIT-compiled model.
        device: The device to load the model onto, default=cuda.
    Returns:
        The JIT compiled model loaded to device and on eval mode.
    """
    if tokenizer_config is None:
        return load_jit_model(jit_filepath, device)
    full_model, ckpts = _load_pytorch_model(jit_filepath, tokenizer_config, device)
    # encoder_model = full_model.encoder_jit()
    # encoder_model.load_state_dict(ckpts.state_dict(), strict=True)
    # return encoder_model.eval().to(device)


def load_decoder_model(
    jit_filepath: str = None,
    tokenizer_config: dict[str, Any] = None,
    device: str = "cuda",
) -> torch.nn.Module | torch.jit.ScriptModule:
    """Loads a torch.nn.Module from a filepath.

    Args:
        jit_filepath: The filepath to the JIT-compiled model.
        device: The device to load the model onto, default=cuda.
    Returns:
        The JIT compiled model loaded to device and on eval mode.
    """
    if tokenizer_config is None:
        return load_jit_model(jit_filepath, device)
    # full_model, ckpts = _load_pytorch_model(jit_filepath, tokenizer_config, device)
    # decoder_model = full_model.decoder_jit()
    # decoder_model.load_state_dict(ckpts.state_dict(), strict=True)
    # return decoder_model.eval().to(device)


# def _load_pytorch_model(
#     jit_filepath: str = None, tokenizer_config: str = None, device: str = "cuda"
# ) -> torch.nn.Module:
#     """Loads a torch.nn.Module from a filepath.

#     Args:
#         jit_filepath: The filepath to the JIT-compiled model.
#         device: The device to load the model onto, default=cuda.
#     Returns:
#         The JIT compiled model loaded to device and on eval mode.
#     """
#     tokenizer_name = tokenizer_config["name"]
#     model = TokenizerModels[tokenizer_name].value(**tokenizer_config)
#     ckpts = torch.jit.load(jit_filepath, map_location=device)
#     return model, ckpts


def load_jit_model(jit_filepath: str = None, device: str = "cuda") -> torch.jit.ScriptModule:
    """Loads a torch.jit.ScriptModule from a filepath.

    Args:
        jit_filepath: The filepath to the JIT-compiled model.
        device: The device to load the model onto, default=cuda.
    Returns:
        The JIT compiled model loaded to device and on eval mode.
    """
    model = torch.jit.load(jit_filepath, map_location=device)
    return model.eval().to(device)


def numpy2tensor(
    input_image: np.ndarray,
    dtype: torch.dtype = _DTYPE,
    device: str = _DEVICE,
    range_min: int = -1,
) -> torch.Tensor:
    """Converts image(dtype=np.uint8) to `dtype` in range [0..255].

    Args:
        input_image: A batch of images in range [0..255], BxHxWx3 layout.
    Returns:
        A torch.Tensor of layout Bx3xHxW in range [-1..1], dtype.
    """
    ndim = input_image.ndim
    indices = list(range(1, ndim))[-1:] + list(range(1, ndim))[:-1]
    image = input_image.transpose((0,) + tuple(indices)) / _UINT8_MAX_F
    if range_min == -1:
        image = 2.0 * image - 1.0
    return torch.from_numpy(image).to(dtype).to(device)


def tensor2numpy(input_tensor: torch.Tensor, range_min: int = -1) -> np.ndarray:
    """Converts tensor in [-1,1] to image(dtype=np.uint8) in range [0..255].

    Args:
        input_tensor: Input image tensor of Bx3xHxW layout, range [-1..1].
    Returns:
        A numpy image of layout BxHxWx3, range [0..255], uint8 dtype.
    """
    if range_min == -1:
        input_tensor = (input_tensor.float() + 1.0) / 2.0
    ndim = input_tensor.ndim
    output_image = input_tensor.clamp(0, 1).cpu().numpy()
    output_image = output_image.transpose((0,) + tuple(range(2, ndim)) + (1,))
    return (output_image * _UINT8_MAX_F + 0.5).astype(np.uint8)


def pad_image_batch(batch: np.ndarray, spatial_align: int = _SPATIAL_ALIGN) -> tuple[np.ndarray, list[int]]:
    """Pads a batch of images to be divisible by `spatial_align`.

    Args:
        batch: The batch of images to pad, layout BxHxWx3, in any range.
        align: The alignment to pad to.
    Returns:
        The padded batch and the crop region.
    """
    height, width = batch.shape[1:3]
    align = spatial_align
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [
        height_to_pad >> 1,
        width_to_pad >> 1,
        height + (height_to_pad >> 1),
        width + (width_to_pad >> 1),
    ]
    batch = np.pad(
        batch,
        (
            (0, 0),
            (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
            (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)),
            (0, 0),
        ),
        mode="constant",
    )
    return batch, crop_region

def unpad_image_batch(batch: np.ndarray, crop_region: list[int]) -> np.ndarray:
    """Unpads image with `crop_region`.

    Args:
        batch: A batch of numpy images, layout BxHxWxC.
        crop_region: [y1,x1,y2,x2] top, left, bot, right crop indices.

    Returns:
        np.ndarray: Cropped numpy image, layout BxHxWxC.
    """
    assert len(crop_region) == 4, "crop_region should be len of 4."
    y1, x1, y2, x2 = crop_region
    return batch[..., y1:y2, x1:x2, :]
