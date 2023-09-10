import os
import subprocess

_mcfost_bin = "mcfost"

def run(filename, options="", delete_previous=False, notebook=False, logfile=None, silent=False):

    if not isinstance(filename, str):
        raise TypeError("First argument to run must be a filename.")
    filename = os.path.normpath(os.path.expanduser(filename))
    if not os.path.exists(filename):
        raise IOError(filename+" does not exist")

    # Finding root directory
    root_dir = "."
    opts = options.split()
    for i, option in enumerate(opts):
        if option == "-root_dir":
            root_dir = opts[i+1]
            break

    if delete_previous:
        subprocess.call("rm -rf "+root_dir+"/data_*", shell = True)

    if silent:
        if logfile is None:
            logfile = "mcfost.log"

    # Saving output in a log file
    if logfile is not None:
        options+= " > "+logfile

    if not silent:
        print("pymcfost: Running mcfost ...")

    r = subprocess.run(_mcfost_bin+" "+filename+" "+options, shell = True)
    if r.returncode:
        raise OSError("mcfost did not run as expected, check mcfost's output")

    if not silent:
        print("pymcfost: Done")
