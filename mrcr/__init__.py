"""MRCR-style marker-retrieval datagen over FinePDFs passages (Phase 2c)."""

from .mrcr import MRCRTask, build_pool, load_markers, main, parse_markers

__all__ = ["MRCRTask", "build_pool", "load_markers", "main", "parse_markers"]
