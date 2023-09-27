import pytest
from pymcfost.parameters import _word_to_bool



os.system('wget http://www.physics.usyd.edu.au/~zwan0382/pymcfost/testdata.tar')
os.system('tar -xvf testdata.tar -C ./tests/')

true_strings = ["True", ".True.", "TRUE", ".TRUE.", "true", ".true.", "t", ".t.", "T", ".T."]
@pytest.mark.parametrize("string", true_strings)
def test_read_true_string(string):
    assert _word_to_bool(string)

false_strings = ["False", ".False.", "FALSE", ".FALSE.", "false", ".false.", "f", ".f.", "F", ".F."]
@pytest.mark.parametrize("string", false_strings)
def test_read_false_string(string):
    assert not _word_to_bool(string)
