import os
import subprocess

_mcfost_bin = "~/mcfost/src/mcfost"

def run(filename, options="", delete_previous=False, notebook=False):

    if not isinstance(filename, str):
        raise TypeError("First argument to run must be a filename.")
    filename = os.path.normpath(os.path.expanduser(filename))
    if not os.path.exists(filename):
        raise IOError(filename+" does not exist")

    if delete_previous:
        subprocess.call("rm -rf data_* ", shell = True)

    print("pymcfost: Running mcfost ...")
    r = subprocess.run(_mcfost_bin+" "+filename+" "+options, shell = True)
    if r.returncode:
        raise OSError("mcfost did not run as expected, check mcfost's output")
    print("pymcfost: Done")
