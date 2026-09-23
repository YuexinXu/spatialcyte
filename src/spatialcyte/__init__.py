"""spatialcyte — multiplexed IHC images to spatial single-cell objects.

Lightweight top-level: only the config types are imported eagerly (no numpy /
image deps) so ``import spatialcyte`` stays cheap. Heavier modules
(``spatialcyte.io``, ``.segment``, ``.quantify``) are imported on use.
"""

from spatialcyte.config import Channel, PanelConfig

__version__ = "0.1.0"
__all__ = ["Channel", "PanelConfig", "__version__"]
