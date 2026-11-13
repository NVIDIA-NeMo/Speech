# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_CONV_AND_LIST_CUDNN = """
import json, os, torch
conv = torch.nn.Conv2d(16, 32, 3, padding=1).cuda()
x = torch.randn(2, 16, 64, 64, device='cuda', requires_grad=True)
y = conv(x)
y.sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
with open('/proc/self/maps') as f:
    paths = {line.split()[-1] for line in f if 'libcudnn' in line}
print(json.dumps(sorted(os.path.realpath(p) for p in paths)))
"""


@pytest.mark.gpu
@pytest.mark.skipif(not Path('/proc/self/maps').exists(), reason='Requires Linux /proc')
@pytest.mark.skipif(not torch.cuda.is_available() or not torch.backends.cudnn.is_available(), reason='Requires cuDNN')
def test_cudnn_convolution_loads_a_single_cudnn_install():
    """cuDNN sublibraries from a second install (e.g. a system cuDNN next to the wheel) cause version mismatches."""
    result = subprocess.run([sys.executable, '-c', _CONV_AND_LIST_CUDNN], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    paths = json.loads(result.stdout.strip().splitlines()[-1])
    assert paths, 'No cuDNN library was mapped during a cuDNN convolution'
    assert len({str(Path(p).parent) for p in paths}) == 1, f'cuDNN loaded from multiple locations: {paths}'
