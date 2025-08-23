# run_optuna.py
import optuna
import pymcfost
import numpy as np
import os
import yaml
from chi2_functions import compute_combined_chi2, validate_parameters

# Load your observed data and noise once
DATA_CUBE_PATH = "alma_cube.fits"
NOISE_STD = 0.001  # replace with real value or dynamic estimate

# SED data configuration
SED_DATA_PATH = "observed_sed.txt"  # path to observed SED data file
SED_NOISE_FACTOR = 0.1  # fractional uncertainty for SED points (10% default)
USE_SED = True  # set to False to disable SED fitting
USE_ALMA = True  # set to False to disable ALMA fitting

# Load parameter definitions from YAML
with open("parameter_config.yaml") as f:
    param_defs = yaml.safe_load(f)["parameters"]


def run_model(params, output_dir):
    # Load the base parameter file
    from pymcfost.parameters import Params, find_parameter_file
    
    # Find the parameter file in the current directory or specify path
    try:
        param_file = find_parameter_file(".")
    except ValueError:
        # If no parameter file found, you might want to specify a default one
        param_file = "ref4.0.para"  # or your default parameter file
    
    # Load parameters
    mcfost_params = Params(param_file)
    
    # Update parameters using the update_parameter_file function
    mcfost_params.update_parameter_file(**params)
    
    # Write the updated parameter file to the output directory
    updated_param_file = os.path.join(output_dir, "updated_params.para")
    mcfost_params.writeto(updated_param_file)
    
    # Run MCFOST with the updated parameter file
    # You might need to adjust this based on your MCFOST setup
    import subprocess
    cmd = f"mcfost {updated_param_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=output_dir)
    
    if result.returncode != 0:
        raise RuntimeError(f"MCFOST failed: {result.stderr}")
    
    # Look for the output fits file (ALMA/line data)
    model_fits_path = None
    possible_fits_paths = [
        os.path.join(output_dir, "data_CO", "lines.fits.gz"),
        os.path.join(output_dir, "data_1.0", "RT.fits.gz"),
        os.path.join(output_dir, "data_1300", "RT.fits.gz"),
    ]
    for path in possible_fits_paths:
        if os.path.exists(path):
            model_fits_path = path
            break
    
    # Look for SED output file
    model_sed_path = None
    possible_sed_paths = [
        os.path.join(output_dir, "data_th", "sed_rt.fits.gz"),
        os.path.join(output_dir, "data_th", "sed_mc.fits.gz"),
        os.path.join(output_dir, "SED.fits.gz"),
    ]
    for path in possible_sed_paths:
        if os.path.exists(path):
            model_sed_path = path
            break
    
    # Check if we have at least one output file
    if model_fits_path is None and model_sed_path is None:
        raise FileNotFoundError(f"No output files found in {output_dir}")
    
    # Use combined chi² function
    return compute_combined_chi2(
        model_fits_path=model_fits_path,
        model_sed_path=model_sed_path,
        data_cube_path=DATA_CUBE_PATH,
        sed_data_path=SED_DATA_PATH,
        noise_std=NOISE_STD,
        use_alma=USE_ALMA,
        use_sed=USE_SED
    )


def objective(trial):
    trial_id = trial.number
    output_dir = f"trials/trial_{trial_id:05d}"
    os.makedirs(output_dir, exist_ok=True)

    params = {}
    for name, definition in param_defs.items():
        if definition["type"] == "float":
            if definition.get("log", False):
                params[name] = trial.suggest_float(name, definition["low"], definition["high"], log=True)
            else:
                params[name] = trial.suggest_float(name, definition["low"], definition["high"])
        elif definition["type"] == "int":
            params[name] = trial.suggest_int(name, definition["low"], definition["high"])
        else:
            raise ValueError(f"Unsupported parameter type: {definition['type']}")

    try:
        # Validate parameters before running the model
        validate_parameters(params)
        
        chi2 = run_model(params, output_dir)
        print(f"Trial {trial_id} completed successfully with chi2 = {chi2:.6f}")
    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        # Log the failed parameters for debugging
        with open(os.path.join(output_dir, "failed_params.txt"), "w") as f:
            f.write(f"Failed parameters:\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nError: {e}\n")
        return float("inf")

    return chi2


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_alma.db",
        study_name="alma_fit",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100, n_jobs=1)  # Each SLURM task runs this,  # or >1 for local parallel




#----- initial version
def objective(trial):
    # Suggest parameters
    params = {
        "inclination": trial.suggest_float("inclination", 0, 90),
        "mass": trial.suggest_float("mass", 1e-4, 1e-1, log=True),
        "scale_height": trial.suggest_float("scale_height", 5, 20),
        # ... repeat for your 15 parameters ...
    }

    # Set up pymcfost model (update parameters in-place or via template)
    model = pymcfost.Model('your_model_name')  # or your wrapper
    model.set_parameters(**params)             # assumes a helper function

    # Run MCFOST
    model.run()

    # Compute score
    model_fits = 'path/to/mcfost/output.fits'
    data_fits = 'path/to/alma_cube.fits'
    noise_std = 0.001  # or from header / estimation
    chi2 = compute_chi2(model_fits, data_fits, noise_std)

    return chi2
