"""
=========================================================
Dataset Metadata
=========================================================
"""

from dataclasses import dataclass


@dataclass
class DatasetInformation:

    name: str

    samples: int

    features: int

    classes: int