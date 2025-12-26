
# Role D: Integration of TE-SI-TR
import torch.nn as nn
from .temporal import TemporalModule
from .spatial import SpatialModule

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = TemporalModule()
        self.spatial = SpatialModule()
