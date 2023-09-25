# import os
#
# working_dir = os.getcwd()


from pathlib import Path
import os
from copy import copy

import pytest
from pymcfost import Params

test_dir = Path(__file__).parent

output_dir = test_dir / "artifacts"
input_dir = test_dir / "corpus"
corpus = list(input_dir.glob("*.para"))

print(output_dir)
print(input_dir)
print(corpus)



