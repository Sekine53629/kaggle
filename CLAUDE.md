# CLAUDE.md - Kaggle Competition Project

## Overview

This is a multi-competition Kaggle project. Each competition is managed as a separate sub-project within the `competitions/` directory.

---

## MANDATORY: One Question at a Time Rule

**CRITICAL**: When creating competitions, notebooks, or making significant changes, Claude MUST:

1. **Ask ONE question at a time** - Do not bundle multiple questions together
2. **Wait for the user's answer** before proceeding to the next question
3. **Confirm before execution** - Always confirm before creating files or making changes

### Example Workflow

```
Claude: "What is the competition name (URL slug)?"
User: "titanic"

Claude: "What is the competition URL?"
User: "https://www.kaggle.com/competitions/titanic"

Claude: "What is the evaluation metric?"
User: "Accuracy"

Claude: "I will create the competition directory at competitions/titanic/. Proceed? (yes/no)"
User: "yes"

[Claude creates the directory]
```

### Questions to Ask for New Competition

1. Competition name (URL slug)
2. Competition URL
3. Evaluation metric
4. Problem type (classification/regression/other)
5. Confirmation to proceed

---

## Directory Structure

```
kaggle/
├── CLAUDE.md                        # This file (project rules)
├── presentation_materials_guide.md  # Notebook creation order & verification gates
├── methodology.md                   # EDA/modeling methodology & verification know-how
├── requirements.txt                 # Shared dependencies
├── .gitignore
├── src/                             # Shared utilities
│   └── utils/
│       └── __init__.py
├── templates/                       # Template notebooks (copy to new competition)
│   ├── template_eda.ipynb
│   └── template_modeling.ipynb
└── competitions/                    # Competition sub-projects (ISOLATED)
    ├── _template/                   # Template for new competitions
    ├── comp_name_1/
    ├── comp_name_2/
    └── ...
```

**IMPORTANT**: Do NOT create `data/`, `input/`, or `output/` folders in the project root. Each competition has its own isolated data folders.

## CRITICAL: Competition Isolation Rules

### Rule 1: One Directory Per Competition

Each competition MUST have its own isolated directory under `competitions/`:

```
competitions/
├── titanic/               # Titanic competition
├── house-prices/          # House Prices competition
├── amex-default/          # Amex Default competition
└── ...
```

### Rule 2: Competition Directory Structure

Each competition directory MUST follow this structure:

```
competitions/{comp_name}/
├── README.md              # Competition info, approach, results
├── input/                 # Competition-specific data (gitignored)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── output/                # Submissions and predictions (gitignored)
│   └── submission_v1.csv
├── notebooks/             # All notebooks for this competition
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_lgbm.ipynb
│   └── ...
├── src/                   # Competition-specific code
│   ├── features.py
│   ├── models.py
│   └── ...
└── models/                # Saved models (gitignored)
```

### Rule 3: NEVER Mix Competition Files

- **DO NOT** put notebooks from different competitions in the same folder
- **DO NOT** share data files between competitions (copy if needed)
- **DO NOT** use generic filenames like `train.csv` in root directories
- **ALWAYS** prefix or organize by competition name

### Rule 4: Naming Conventions

Competition directories should match the Kaggle URL slug:
- URL: `kaggle.com/competitions/titanic` → Directory: `competitions/titanic/`
- URL: `kaggle.com/competitions/house-prices-advanced-regression-techniques` → Directory: `competitions/house-prices/`

Notebooks should be numbered for execution order:
- `01_eda.ipynb`
- `02_feature_engineering.ipynb`
- `03_baseline_model.ipynb`
- `04_feature_selection.ipynb`
- `05_final_model.ipynb`

### Rule 5: Path References in Notebooks

Always use relative paths from the notebook location:

```python
# In competitions/{comp_name}/notebooks/01_eda.ipynb
INPUT_DIR = Path('../input')
OUTPUT_DIR = Path('../output')

# To use shared utilities
import sys
sys.path.append('../../..')  # Go to project root
from src.utils import seed_everything
```

### Rule 6: Competition README Template

Each competition MUST have a README.md:

```markdown
# Competition Name

- URL: https://www.kaggle.com/competitions/xxx
- Start: YYYY-MM-DD
- End: YYYY-MM-DD
- Evaluation Metric: RMSE / AUC / etc.

## Approach

1. EDA findings
2. Feature engineering strategy
3. Model selection

## Results

| Version | Model | CV Score | LB Score | Notes |
|---------|-------|----------|----------|-------|
| v1      | LGBM  | 0.xxx    | 0.xxx    | Baseline |

## Key Learnings

- ...
```

## Creating a New Competition

Run this to create a new competition structure:

```bash
# Replace {comp_name} with actual competition name
mkdir -p competitions/{comp_name}/{input,output,notebooks,src,models}
touch competitions/{comp_name}/README.md
touch competitions/{comp_name}/src/__init__.py
```

Or copy the template:

```bash
cp -r competitions/_template competitions/{comp_name}
```

## Shared Utilities

The `src/utils/` module contains shared code:
- `seed_everything(seed)` - Set random seed
- `reduce_mem_usage(df)` - Reduce DataFrame memory

Import in competition notebooks:
```python
import sys
sys.path.append('../../..')
from src.utils import seed_everything, reduce_mem_usage
```

## Git Workflow

### What to Commit
- All code (.py, .ipynb)
- README.md files
- requirements.txt
- CLAUDE.md

### What NOT to Commit (gitignored)
- Data files (*.csv, *.parquet, *.feather)
- Model files (*.pkl, *.joblib, *.h5, *.pth)
- Outputs and submissions
- Kaggle API credentials

## Document Reference Order

When starting or resuming work on a competition, Claude MUST read these documents in order:

| Priority | Document | Purpose |
|----------|----------|---------|
| 1 | `CLAUDE.md` (this file) | Project structure, isolation rules, naming conventions |
| 2 | `presentation_materials_guide.md` | Notebook creation order, verification gates, intermediate data rules |
| 3 | `methodology.md` | EDA/modeling phases, verification know-how, reusable code patterns |
| 4 | `competitions/{comp_name}/README.md` | Competition-specific info, approach, results |
| 5 | `presentation_materials_guide.md` status table | Current progress for the competition |

### When to Re-read

- **Starting a new session**: Always read #1 and #2
- **Creating a new notebook**: Read #2 (verification gates) and #3 (methodology phases)
- **Resuming after interruption**: Read #2 status table first, then #5

---

## Compliance Rules

### Rule C1: Verification Gates Are Mandatory

Every notebook step in `presentation_materials_guide.md` has a verification gate table. Claude MUST:
- Run through each check in the gate before marking a step as complete
- NOT proceed to the next step if any gate check fails
- Document which checks passed/failed when reporting status to the user

### Rule C2: confirmed_settings Propagation

The `02_feature_design.pkl` file contains a `confirmed_settings` dict that defines:
- `best_train_start`: Training data start date
- `best_nan_strategy`: NaN handling approach
- `best_rolling_config`: Rolling feature calculation method

**All downstream notebooks (03-x, 04) MUST load and apply these settings.** Never hardcode these values.

### Rule C3: Intermediate Data Chain Integrity

```
01_eda_results.pkl → 02_feature_design.pkl → 03-x_results.pkl → 04_comparison_results.pkl
```

- Each file depends on all upstream files
- If an upstream file is regenerated, ALL downstream files must be regenerated
- Before creating any notebook, verify its upstream pickle files exist

### Rule C4: Notebook Path References

Notebooks in `notebooks/説明用資料/` are 2 levels deep from the competition root:

```python
# CORRECT (from 説明用資料/ directory)
INPUT_DIR = Path('../../input')
OUTPUT_DIR = Path('../../output')
INTERMEDIATE_DIR = Path('./intermediate')

# WRONG (only 1 level)
INPUT_DIR = Path('../input')  # This will fail!
```

### Rule C5: Font Configuration (Windows)

Use `MS Gothic` for Japanese text in matplotlib. `IPAGothic` is not available on Windows.

```python
plt.rcParams['font.family'] = 'MS Gothic'
```

### Rule C6: Python Version Awareness

The project uses Python 3.13. Pickle files saved with Python 3.13 cannot be loaded by older Python versions (e.g., 3.8). Always use the same Python version for all notebook execution.

---

## Verification Know-How Reference

Detailed verification patterns are documented in `methodology.md` (Section: Verification Know-How). Key patterns:

| ID | Pattern | When to Use |
|----|---------|-------------|
| V1 | Feature Leakage Detection | After creating features (Step 2) |
| V2 | Temporal Leakage in Rolling | After creating rolling features (Step 2) |
| V3 | CV-LB Alignment Check | After each submission |
| V4 | Prediction Sanity Check | After each model training (Step 3) |
| V5 | NaN Strategy Validation | During feature design (Step 2, Section 11) |
| V6 | Rolling Window Config Test | During feature design (Step 2, Section 12) |
| V7 | Ensemble Weight Stability | During comparison (Step 4) |
| V8 | Business vs Calendar Rolling | During feature design (Step 2, Section 12) |

---

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download competition data (requires kaggle.json)
kaggle competitions download -c {comp_name} -p competitions/{comp_name}/input

# Submit prediction
kaggle competitions submit -c {comp_name} -f competitions/{comp_name}/output/submission.csv -m "message"
```
