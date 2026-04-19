"""Self-contained resnet1d_wang reimplementation.

Matches the benchmark configuration used by Strodthoff 2021 / Wang 2017:
    ResNet1d(BasicBlock1d, [1, 1, 1], inplanes=128, kernel_size=[5, 3],
             kernel_size_stem=7, stride_stem=1, pooling_stem=False)

Structure: stem (k=7, stride=1, no pool) -> 3 stages of 1 BasicBlock each,
all 128 channels; stage 2 and 3 downsample by stride-2 residual. Head uses
adaptive avg+max pool (concat), dropout, linear to num_classes. This preserves
the benchmark macro-AUC of 0.930 on the diagnostic superclass task.

Hooks for SAE extraction are registered at the output of each of the three
residual stages (stage1, stage2, stage3). All three outputs have channel
dimension 128.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(cin, cout, k=3, stride=1):
    return nn.Conv1d(cin, cout, kernel_size=k, stride=stride,
                     padding=(k - 1) // 2, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, k=(5, 3), stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv(cin, cout, k=k[0], stride=stride)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = _conv(cout, cout, k=k[1])
        self.bn2 = nn.BatchNorm1d(cout)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class ConcatPool1d(nn.Module):
    def forward(self, x):
        return torch.cat([F.adaptive_avg_pool1d(x, 1),
                          F.adaptive_max_pool1d(x, 1)], dim=1).flatten(1)


class ResNet1dWang(nn.Module):
    """resnet1d_wang: 3 stages x 1 BasicBlock, 128 channels, stages 2-3 stride-2."""

    def __init__(self, num_classes: int = 5, input_channels: int = 12,
                 inplanes: int = 128, k=(5, 3), k_stem: int = 7,
                 head_dropout: float = 0.5):
        super().__init__()
        self.inplanes = inplanes
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, inplanes, kernel_size=k_stem,
                      stride=1, padding=(k_stem - 1) // 2, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(inplanes, stride=1, k=k)
        self.stage2 = self._make_stage(inplanes, stride=2, k=k)
        self.stage3 = self._make_stage(inplanes, stride=2, k=k)
        self.pool = ConcatPool1d()
        self.head = nn.Sequential(
            nn.Linear(2 * inplanes, inplanes),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(inplanes, num_classes),
        )

    def _make_stage(self, planes, stride, k):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        blk = BasicBlock1d(self.inplanes, planes, k=k, stride=stride,
                           downsample=downsample)
        self.inplanes = planes
        return blk

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.head(x)

    def register_hooks(self):
        """Return {name: list-of-captured-tensors} dict populated on forward."""
        store: dict[str, list[torch.Tensor]] = {"stage1": [], "stage2": [], "stage3": []}

        def mk(name):
            def hook(_m, _i, out):
                store[name].append(out.detach())
            return hook

        self.stage1.register_forward_hook(mk("stage1"))
        self.stage2.register_forward_hook(mk("stage2"))
        self.stage3.register_forward_hook(mk("stage3"))
        return store
