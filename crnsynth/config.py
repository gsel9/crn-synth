"""Project configuration settings"""
import os
from pathlib import Path

# Paths
PATH_REPO = Path(os.path.dirname(os.path.realpath(__file__))).parent
PATH_RESULTS = PATH_REPO / "results"

PATH_SYNTH_CONFIGS = PATH_REPO / "synthesis_configs"

DEFAULT_DIRS = ["synthetic_data", "generators", "configs", "reports"]

# path to data for TSD
PATH_DATA_TSD = (
    PATH_REPO.parents[2]
    if PATH_REPO.parents[2].name == "durable"
    else PATH_REPO.parents[3]
)
PATH_LUNG_DATA_TSD = PATH_DATA_TSD / "lung-cancer/synthetic/data"

# path to all data files available for synthesis.
PATH_DATA = {
    "adult": PATH_REPO / "data/adult.csv",
}

PATH_GENERATOR_CONFIG = {
    "aim": PATH_SYNTH_CONFIGS / "config_aim.json",
    "decaf": PATH_SYNTH_CONFIGS / "config_decaf.json",
    "adsgan": PATH_SYNTH_CONFIGS / "config_adsgan.json",
    "ctgan": PATH_SYNTH_CONFIGS / "config_ctgan.json",
    "ddpm": PATH_SYNTH_CONFIGS / "config_ddpm.json",
    "dpgan": PATH_SYNTH_CONFIGS / "config_dpgan.json",
    "pategan": PATH_SYNTH_CONFIGS / "config_pategan.json",
    "rtvae": PATH_SYNTH_CONFIGS / "config_rtvae.json",
    "survival_ctgan": PATH_SYNTH_CONFIGS / "config_survival_ctgan.json",
    "survival_nflow": PATH_SYNTH_CONFIGS / "config_survival_nflow.json",
}
