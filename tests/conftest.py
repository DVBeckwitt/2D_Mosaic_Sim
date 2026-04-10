from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _avoid_plotly_pyarrow_probe():
    pyarrow_module = sys.modules.pop("pyarrow", None)
    try:
        yield
    finally:
        if pyarrow_module is not None:
            sys.modules["pyarrow"] = pyarrow_module
