from pathlib import Path
import os

import pytest
from pymcfost import Params

test_dir = Path(__file__).parent

output_dir = test_dir / "artifacts"
input_dir = test_dir / "corpus"
corpus = list(input_dir.glob("*.para"))


@pytest.mark.parametrize("parafile", corpus)
def test_io_copy(parafile):
    """Read a valid .para file, check that it can still be read and written with no data corruption"""
    os.makedirs(output_dir, exist_ok=True)
    p1 = Params(str(parafile))
    output_file = str(output_dir / "_".join(["copy", parafile.name]))
    p1.writeto(output_file)
    p2 = Params(output_file)

    assert str(p1) == str(p2)
