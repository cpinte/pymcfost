import os
import subprocess

def run(filename, options="", delete_previous=False):

    if not isinstance(filename, str):
        raise TypeError("First argument to run must be a filename.")
    filename = os.path.normpath(os.path.expanduser(filename))
    if not os.path.exists(filename):
        raise IOError(filename+" does not exist")

    if delete_previous:
        subprocess.call("rm -rf data_* ", shell = True)



    subprocess.call("mcfost "+filename+" "+options, shell = True)
