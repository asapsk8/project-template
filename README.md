# End-to-end regression pipeline (scikit-learn Diabetes)

This repo is a minimal, reproducible data-science workflow:
- load a public dataset (sklearn Diabetes)
- train 2 models (Ridge, Random Forest)
- evaluate with RMSE on a held-out test set
- save artifacts to `reports/` (residual plot + results)

## How to run (PyCharm)
1. Select interpreter: `.venv123`
2. Install deps: numpy, pandas, matplotlib, scikit-learn, pyyaml, pytest
3. Run `run.py`
4. Run tests in `tests/`

## Outputs
After running `run.py`:
- `reports/residuals.png`
- `reports/results.txt`

## Notes
RMSE is computed as sqrt(MSE) to avoid scikit-learn version differences.
