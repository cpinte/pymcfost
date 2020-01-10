from pathlib import Path
import os
from copy import copy

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
    s1 = copy(str(p1))

    output_file = str(output_dir / "_".join(["copy", parafile.name]))
    p1.writeto(output_file)
    p2 = Params(output_file)
    s2 = str(p2)

    s3 = str(p1)
    assert s3 == s1
    assert s2 == s1


@pytest.mark.parametrize(
    "parafile,expected",
    [
        ("ref3.0.para", (False, True, True, True)),
        ("ref3.0_switch_bools.para", (True, False, False, False)),
    ],
)
def test_bool_parsing(parafile, expected):
    p = Params(input_dir / parafile)
    result = (
        p.simu.radial_migration,
        p.simu.image_symmetry,
        p.mol.compute_pop,  # lpop
        p.stars[0].is_bb,
    )
    assert result == expected
