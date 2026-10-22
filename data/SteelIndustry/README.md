# Steel Industry Energy Consumption

`Steel_industry_data.csv` is the unchanged CSV from the
[UCI dataset](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption),
[DOI 10.24432/C52G8C](https://doi.org/10.24432/C52G8C), by Sathishkumar V E,
Changsun Shin and Yongyun Cho (2021). License:
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Downloaded from UCI's ZIP on 22 October 2026. CSV SHA-256:
`9b1cee6f9cb9cd9df2b95814ca90a9a2ff15b7f5f1fba0fae3c643e82072eacc`.

`SteelIndustryDescriptor` reads all 35,040 rows in source order. The target is
`Usage_kWh`; the nine other predictors are used, with `date` omitted. WeekStatus
is Weekend=0, Weekday=1; weekdays are Monday=0 through Sunday=6; Load_Type is
Light_Load=0, Medium_Load=1, Maximum_Load=2. Punctuation in two GP variable names
is replaced by underscores. No lags, scaling, resampling or new scoring logic
are introduced. The existing HPO evaluation behavior is unchanged.

The raw predictors include contemporaneous electrical measurements and CO2;
their suitability for the final paper (particularly CO2 as a target proxy)
remains part of benchmark-protocol design. This loader does not define a
forecasting task or provide a held-out performance claim.

The CSV is copied beside the executable under `data/SteelIndustry/`. An explicit
CSV path can also be passed to the descriptor constructor.

```powershell
dotnet run --project HyperOp.Experiments -- --algorithm RandomSearch --instance SteelIndustry --configurations 1 --evaluations 1000
```
