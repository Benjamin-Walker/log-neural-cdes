from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Dataset:
    name: str
    raw_dataloaders: Dict[str, Any]
    sig_dataloaders: Dict[str, Any]
    data_dim: int
    label_dim: int
