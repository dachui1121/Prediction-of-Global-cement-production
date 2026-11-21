# Cement production and residential floor-area projections

Python 3.13. The code uses one separately supplied English data file, `Research_Data.xlsx`.

Place it in `data/Research_Data.xlsx`, or specify its path with `--workbook`. The data file contains all original model input versions, observed floor-area anchors, global scaling and emissions calculations, and reference series.

```bash
python -m pip install -r requirements.txt
python run.py --list
python run.py --workbook Research_Data.xlsx --check-inputs
python run.py --workbook Research_Data.xlsx --output-dir results
python run_floor_area.py --workbook Research_Data.xlsx --output-dir floor_area_results
```

Select countries and roles:

```bash
python run.py --scenario bau --country united_states --role nonrep --workbook Research_Data.xlsx
python run_floor_area.py --country china --workbook Research_Data.xlsx
```

BAUS includes 11 countries and 22 cement models. DAS includes 10 countries and 20 cement models; the command aliases are `bau`/`baus` and `ds`/`das`. Eleven independent BAUS floor-area fitting routines are included.

The US replacement model is `warmup_1989`; the non-replacement model is `nonreplacement`. Non-replacement historical output uses the corresponding replacement fit for 1990-2024. Its original calibration and forecast calculations for 2025-2100 are retained.

The `Input Index` and `Data Dictionary` identify each input worksheet and its numerical units. The input adapter restores the original field names and dtypes in memory; no numerical model modules are changed. Unused trailing rows with no year are omitted.

The US, Indonesia, Brazil and India floor-area fitting inputs differ from the locked cement inputs. Both recorded input versions are included. Running a floor-area fit does not overwrite the cement input series.

Verify all outputs against the references inside the same data file:

```bash
python -m unittest discover -s tests -v
python tests/verify_workbook_results.py --workbook Research_Data.xlsx --cement-results results --floor-results floor_area_results
```

The workbook and code are matched by SHA-256. A different input version is rejected before execution. Input data and numerical results are excluded from the public code archive.
