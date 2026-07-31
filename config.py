"""
==========================================================
ATC-HGWO-PSO Configuration File
==========================================================

Author : Viraj Mohod
Journal : Emerald
Project : Adaptive Tent Chaotic Hybrid GWO-PSO

==========================================================
"""

from pathlib import Path

# ======================================================
# PATHS
# ======================================================

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "datasets"

RAW_DATA = DATA_DIR / "raw"

PROCESSED_DATA = DATA_DIR / "processed"

OUTPUT_DIR = ROOT_DIR / "outputs"

FIGURE_DIR = OUTPUT_DIR / "figures"

CSV_DIR = OUTPUT_DIR / "csv"

LATEX_DIR = OUTPUT_DIR / "latex"

LOG_DIR = OUTPUT_DIR / "logs"

# ======================================================
# RANDOM SEED
# ======================================================

RANDOM_STATE = 42

# ======================================================
# DATASET PARAMETERS
# ======================================================

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.10

NORMALIZATION = "minmax"

# ======================================================
# OPTIMIZER PARAMETERS
# ======================================================

POPULATION_SIZE = 30

MAX_ITERATIONS = 100

DIMENSIONS = None

ALPHA = 0.99

# ======================================================
# PSO
# ======================================================

W_MAX = 0.9

W_MIN = 0.4

C1 = 2.0

C2 = 2.0

# ======================================================
# TENT MAP
# ======================================================

MU = 0.7

# ======================================================
# CLASSIFIER
# ======================================================

KNN_NEIGHBOURS = 5

CROSS_VALIDATION = 10

# ======================================================
# SAVE DIRECTORIES
# ======================================================

for folder in [

    OUTPUT_DIR,

    FIGURE_DIR,

    CSV_DIR,

    LATEX_DIR,

    LOG_DIR,

]:

    folder.mkdir(parents=True, exist_ok=True)