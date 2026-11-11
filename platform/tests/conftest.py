import importlib.util
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
spec = importlib.util.spec_from_file_location(
    "annotation_app",
    APP / "__init__.py",
    submodule_search_locations=[str(APP)],
)
module = importlib.util.module_from_spec(spec)
sys.modules["annotation_app"] = module
spec.loader.exec_module(module)
