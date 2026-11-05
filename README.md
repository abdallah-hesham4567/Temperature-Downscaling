# Temperature Downscaling: Gradient Boosting vs Neural Networks

<p align="center">
  <img src="https://img.shields.io/badge/LANGUAGE-PYTHON-3776AB?style=for-the-badge&labelColor=555555&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FRAMEWORK-PYTORCH-EE4C2C?style=for-the-badge&labelColor=555555&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/BOOSTING-XGBOOST-1B8A5A?style=for-the-badge&labelColor=555555" alt="XGBoost">
  <img src="https://img.shields.io/badge/BOOSTING-CATBOOST-FFCC00?style=for-the-badge&labelColor=555555" alt="CatBoost">
  <img src="https://img.shields.io/badge/ML-SCIKIT--LEARN-F7931E?style=for-the-badge&labelColor=555555&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/ARRAYS-NUMPY-4DABCF?style=for-the-badge&labelColor=555555&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/TABULAR-PANDAS-150458?style=for-the-badge&labelColor=555555&logo=pandas&logoColor=white" alt="pandas">
  <img src="https://img.shields.io/badge/GEOSPATIAL-XARRAY-3B6EA5?style=for-the-badge&labelColor=555555" alt="xarray">

Predicting temperature on a finer spatial grid from coarse weather data, and comparing five models under a spatially held-out evaluation.

**Models compared:** Random Forest, XGBoost, CatBoost, MLP (PyTorch), CNN (PyTorch), each against a plain-interpolation baseline.

> **Scope note:** this repo uses a *proxy* dataset (NCEP/NCAR reanalysis, 2.5° grid) where the coarse input is made by coarsening the target. It validates the pipeline and the evaluation protocol. It is not a street-level model yet. See [Limitations](#limitations) and [Roadmap](#roadmap).

---

## The problem

Weather forecasts and reanalyses come on coarse grids (tens of kilometres). Temperature inside one grid cell is not uniform: it depends on elevation, land cover, urban fraction, and time of day. Statistical downscaling learns that correction.

```text
Coarse temperature grid
        +
Location + time features        ->   ML model   ->   Fine-resolution temperature
        (later: elevation, land cover, ...)
```

## Approach

```text
air_temperature.nc (xarray)
        |
        |  coarsen 2x2 mean            2.5° -> 5°     (model INPUT)
        v
coarse grid --bilinear interpolation--> back on 2.5° grid   (= BASELINE)
        |
        |  target = fine - interpolated coarse           (the RESIDUAL)
        v
features table / map tensors
        |
        |  spatial split: hold out whole 5x5-cell blocks
        v
Random Forest | XGBoost | CatBoost | MLP | CNN
        |
        v
prediction = interpolated coarse + predicted residual
        |
        v
RMSE / MAE / R² / train time  ->  comparison table, error map, example maps
```

Key design choices:

- **Predict the residual, not the temperature.** Tree models output piecewise-constant values and cannot extrapolate. Learning only the correction on top of the interpolation is a much easier problem.
- **Spatial hold-out.** Neighbouring grid points are nearly identical, so a random row split leaks. The map is cut into 5x5-cell blocks and 20% of whole blocks (11 of 55) are held out. Every model is scored on the same held-out blocks.
- **Interpolation baseline.** Any model must beat plain interpolation of the coarse grid to be worth using.
- **Same split, same metrics, same target** for all models.

## Data

| Item | Value |
|---|---|
| Source | NCEP/NCAR reanalysis, `air_temperature` (xarray tutorial dataset) |
| Variable | Air temperature, converted from K to °C |
| Period | 2013-01-01 to 2014-12-31, 6-hourly (2,920 time steps) |
| Grid | 25 x 53 points, 2.5°, lat 15–75, lon 200–330 (North America) |
| Model input | 5° grid (2x2 mean of the target), interpolated back to 2.5° |
| Model target | Original 2.5° temperature |

To get the file:

```python
import xarray as xr
xr.tutorial.open_dataset("air_temperature").to_netcdf("air_temperature.nc")   # needs: pip install pooch
```

## Models

| Model | Configuration |
|---|---|
| Random Forest | 100 trees, `min_samples_leaf=50` |
| XGBoost | 300 trees, depth 8, lr 0.1, subsample 0.8, colsample 0.8, `hist` |
| CatBoost | 300 iterations, depth 8, lr 0.1 |
| MLP (PyTorch) | 6 -> 128 -> 128 -> 64 -> 1, ReLU, Adam (lr 1e-2), 20 epochs, batch 1024 |
| CNN (PyTorch) | 3 conv layers (3x3, 32 channels), Adam (lr 1e-3), 20 epochs, loss masked to training pixels |

Features for the tree models and the MLP: interpolated coarse temperature, latitude, longitude, month, hour, day of year. These models train on a random 200k-row subsample of the training blocks. The CNN takes the coarse map as its only input channel and trains on every time step of the training blocks.

## Results

Evaluated on 744,600 rows from 11 held-out blocks (RMSE and MAE in °C).

| Model | RMSE | MAE | R² | Train time (s) | RMSE improvement vs baseline |
|---|---|---|---|---|---|
| Interpolation (baseline) | 1.6646 | 1.1087 | 0.9908 | – | 0% |
| Random Forest | 1.6892 | 1.1434 | 0.9905 | 76.6 | -1.5% |
| XGBoost | 1.6011 | 1.0718 | 0.9915 | 8.5 | 3.8% |
| CatBoost | 1.5487 | 1.0435 | 0.9920 | 11.5 | 7.0% |
| MLP (PyTorch) | 1.7278 | 1.1400 | 0.9900 | 75.3823 | -3.7965% |
| CNN (PyTorch) | 1.3825 | 0.8592 | 0.9936 | 151.9887 | 16.9465% |


Model	RMSE	MAE	R2	Train time (s)	Improvement vs baseline (%)
0	CNN (PyTorch)	1.3825	0.8592	0.9936	151.9887	16.9465
1	CatBoost	1.5487	1.0435	0.9920	10.9721	6.9621
2	XGBoost	1.6011	1.0718	0.9915	7.7601	3.8103
3	Interpolation (baseline)	1.6646	1.1087	0.9908	0.0000	0.0000
4	Random Forest	1.6892	1.1434	0.9905	89.4603	-1.4785
5	MLP (PyTorch)	1.7278	1.1400	0.9900	75.3823	-3.7965

**Findings**

- Among the tree models, CatBoost is best and XGBoost is second. Both beat interpolation on held-out locations.
- Random Forest does not beat the interpolation baseline on unseen blocks, even with a larger `min_samples_leaf`. It fits the training locations but does not carry the correction to new ones.
- Gains over interpolation are modest across the board. This is expected: the proxy has no fine-scale predictors such as elevation, so there is little information beyond location and time for the models to use.
- TODO: one or two sentences on the MLP and CNN once the results are in.

## Repository structure

```text
.
├── temperature_downscaling.ipynb   # full pipeline, runs top to bottom
├── figures/                        # saved plots used in this README
├── requirements.txt
└── README.md
```

## Run it

```bash
pip install xarray netcdf4 numpy pandas matplotlib scikit-learn xgboost catboost torch pooch
```

1. Create `air_temperature.nc` (snippet above).
2. Open `temperature_downscaling.ipynb` in Colab or Jupyter. In section 3, set the path to your copy of the file (the notebook uses `/content/air_temperature.nc`, the Colab default).
3. Run all cells. It takes roughly 10 minutes on a CPU. Seeds are fixed (`seed = 42`).

## Limitations

- **Proxy, not the real task.** Input and target come from the same 2.5° reanalysis, so this tests the method, not street-level accuracy.
- **No fine-scale predictors.** No elevation, land cover, urban fraction, or humidity and wind yet.
- **Unequal comparison conditions.** The CNN sees far more training data than the 200k-row models, and it uses only the coarse map (no lat, lon, or time features).
- **One split, one seed.** Differences of a few hundredths in RMSE are within noise. Repeating over different held-out blocks and seeds is needed before declaring a winner.
- **Spatial hold-out only.** There is no separate time-based hold-out yet.

## Roadmap

1. **Real data:** ERA5 (0.25°) as input with ERA5-Land (0.1°) or weather-station observations as the target, stored and read through Zarr.
2. **Fine-scale features:** elevation from a DEM, land cover, urban fraction, vegetation, distance to water.
3. **CNN with static channels:** feed elevation and land-cover rasters as extra input channels, and try a U-Net.
4. **Stronger evaluation:** repeated spatial cross-validation, a time-based hold-out, and error broken down by region and hour of day.
5. **Code structure:** move the notebook logic into a small package (`data.py`, `features.py`, `models.py`) with tests.

## Skills demonstrated

- Geospatial arrays with **xarray**: coarsening, interpolation, coordinate-based selection, mapping between grid and table layouts
- **Gradient boosting** (XGBoost, CatBoost) and Random Forest on tabular features
- **PyTorch**: an MLP and a fully convolutional network with a spatially masked loss
- Leakage-aware **spatial validation** and a fair multi-model comparison against a non-ML baseline

---

**Author:** Abdallah Hesham · [GitHub](https://github.com/abdallah-hesham4567) · [LinkedIn](https://linkedin.com/in/abdallah-hesham223)
