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
├── CLAUDE.md              # This file
├── requirements.txt       # Shared dependencies
├── .gitignore
├── src/                   # Shared utilities
│   └── utils/
│       └── __init__.py
├── templates/             # Template notebooks (copy to new competition)
│   ├── template_eda.ipynb
│   └── template_modeling.ipynb
└── competitions/          # Competition sub-projects (ISOLATED)
    ├── _template/         # Template for new competitions
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

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download competition data (requires kaggle.json)
kaggle competitions download -c {comp_name} -p competitions/{comp_name}/input

# Submit prediction
kaggle competitions submit -c {comp_name} -f competitions/{comp_name}/output/submission.csv -m "message"
```
