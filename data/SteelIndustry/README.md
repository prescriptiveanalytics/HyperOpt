# Steel Industry Energy Consumption

`Steel_industry_data.csv` contains 35,040 observations from the
[UCI dataset](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
by Sathishkumar V E, Changsun Shin and Yongyun Cho (2021).
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
DOI: [10.24432/C52G8C](https://doi.org/10.24432/C52G8C).

CSV SHA-256: `9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc`.

## Data preparation

The target is `Usage_kWh`. All nine predictors are retained, including CO₂;
`date` is excluded. Rows stay in source order, with chronological partitions:
21,024 training, 7,008 validation and 7,008 test rows.

Categorical values use fixed numeric codes:

- WeekStatus: Weekend=0, Weekday=1.
- Day_of_week: Monday=0 through Sunday=6.
- Load_Type: Light_Load=0, Medium_Load=1, Maximum_Load=2.

GP uses training-fitted linear calibration. HPO minimizes validation NMSE; test data
score the selected expression and fresh-seed runs of the nominated configuration.
All methods and repetitions reuse the same partitions.

The task is contemporaneous regression. The CO₂ and electrical predictors do not
establish availability for advance forecasting.

## Run

```powershell
dotnet run --project HyperOp.Experiments -c Release -- --algorithm RandomSearch --instance SteelIndustry --configurations 10 --evaluations 1000 --output ./results/steel-demo
```

The CSV is copied to the application output automatically.
