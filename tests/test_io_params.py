from pathlib import Path
import os

from pymcfost import Params

test_dir = Path(__file__).parent


def test_io_copy():
    """Read a parafile, write it elsewhere and attempt to read the copy
    Goal : check that the copy is still valid as an input
    """
    input_dir = test_dir / "corpus"
    output_dir = test_dir / "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    for parafile in input_dir.glob("*.para"):
        p = Params(str(parafile))

        output_file = str(output_dir / "_".join(["copy", parafile.name]))
        p.writeto(output_file)
        p2 = Params(output_file)
