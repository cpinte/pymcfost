# run_optuna.py
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pymcfost
import numpy as np
import os
import yaml
from .chi2_functions import compute_combined_chi2, compute_chi2_sed, validate_parameters, ObservedSED
from .run import run as run_mcfost
from .parameters import Params
import copy

"""
We use a class to pass additional arguments at initialisation
"""
class Objective_SED:

    def __init__(self, SED, param_defs, ref_P, options):
        # Reading data and simulation setup only once
        self.observed_SED = SED

        # Storing simulation configuration
        self.param_defs = param_defs
        self.ref_P = ref_P
        self.options = options

    def __call__(self, trial):

        trial_id = trial.number
        output_dir = f"trials/trial_{trial_id:05d}"
        os.makedirs(output_dir, exist_ok=True)

        # Creating trial
        params = {}
        for name, definition in self.param_defs.items():
            if definition["type"] == "float":
                if definition.get("log", False):
                    params[name] = trial.suggest_float(name, definition["low"], definition["high"], log=True)
                else:
                    params[name] = trial.suggest_float(name, definition["low"], definition["high"])
            elif definition["type"] == "int":
                params[name] = trial.suggest_int(name, definition["low"], definition["high"])
            else:
                raise ValueError(f"Unsupported parameter type: {definition['type']}")

        # Run mcfost and computing chi2
        try:
            # Validate parameters before running the model
            validate_parameters(params)

            P = copy.deepcopy(self.ref_P)

            # Update parameters using the update_parameter_file function
            P.update_parameters(params)

            # Write parameter file
            para_file = output_dir+"/optuna.para"
            P.writeto(para_file)

            # run mcfost
            run_mcfost(para_file, options=self.options+" -root_dir "+output_dir, silent=True)

            # compute_chi2
            mcfost_SED = output_dir+"/data_th"
            chi2 = compute_chi2_sed(mcfost_SED, self.observed_SED, i=0, iaz=0)

            print(f"Trial {trial_id} completed successfully with chi2 = {chi2:.6f}")

            return chi2

        except Exception as e:
            print(f"Trial {trial_id} failed: {e}")

            # Log the failed parameters for debugging
            with open(os.path.join(output_dir, "failed_params.txt"), "w") as f:
                f.write(f"Failed parameters:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
                    f.write(f"\nError: {e}\n")

            return float("inf")



def run_optuna_SED(config_file="parameter_config.yaml", n_jobs=1, n_trials=100):

    # Load fitting configution file
    with open(config_file) as f:
        config = yaml.safe_load(f)

    param_defs = config["parameters"]

    # read mcfost reference parameter file and options
    P_ref = Params(config['reference_file'])
    options = config['options']

    # Reading SED data
    SED_file = config['SED_file']
    SED = ObservedSED.from_file(SED_file)

    # Objective function
    objective = Objective_SED(SED, param_defs, P_ref, options)

    # Setup optuna storage for multi-node NFS and study
    storage = JournalStorage(JournalFileBackend("optuna_journal_storage.log"))
    study = optuna.create_study(
        direction="minimize",
        storage=storage,
        study_name="SED_fit",
        load_if_exists=True
    )

    # Run the actual fit
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)  # Each SLURM task runs this,  # or >1 for local parallel

    return


def load_optuna_study(study_name="SED_fit", journal_path="optuna_journal_storage.log"):
    """
    Load an existing Optuna study using the same JournalStorage backend.

    Args:
        study_name (str): Name of the Optuna study to load. Defaults to "SED_fit".
        journal_path (str): Path to the journal storage log file. Defaults to "optuna_journal_storage.log".

    Returns:
        optuna.study.Study: The loaded study instance.
    """
    storage = JournalStorage(JournalFileBackend(journal_path))
    return optuna.load_study(study_name=study_name, storage=storage)


if __name__ == "__main__":
    run_optuna()


#----- initial version
#def objective(trial):
#    # Suggest parameters
#    params = {
#        "inclination": trial.suggest_float("inclination", 0, 90),
#        "mass": trial.suggest_float("mass", 1e-4, 1e-1, log=True),
#        "scale_height": trial.suggest_float("scale_height", 5, 20),
#        # ... repeat for your 15 parameters ...
#    }
#
#    # Set up pymcfost model (update parameters in-place or via template)
#    model = pymcfost.Model('your_model_name')  # or your wrapper
#    model.set_parameters(**params)             # assumes a helper function
#
#    # Run MCFOST
#    model.run()
#
#    # Compute score
#    model_fits = 'path/to/mcfost/output.fits'
#    data_fits = 'path/to/alma_cube.fits'
#    noise_std = 0.001  # or from header / estimation
#    chi2 = compute_chi2(model_fits, data_fits, noise_std)
#
#    return chi2
