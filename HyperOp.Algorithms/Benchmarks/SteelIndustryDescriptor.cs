using System.Globalization;
using CsvHelper;

namespace HyperOp.Algorithms.Benchmarks;

public sealed class SteelIndustryDescriptor : RegressionDescriptor
{
    private readonly string csvPath;
    private static readonly string[] NumericColumns =
    [
        "Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
        "CO2(tCO2)", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "NSM"
    ];

    public SteelIndustryDescriptor(string? csvPath = null)
    {
        this.csvPath = csvPath ?? Path.Combine(AppContext.BaseDirectory,
            "data", "SteelIndustry", "Steel_industry_data.csv");
    }

    public override string Name => "Steel Industry Energy Consumption";
    public override string Description => "UCI Steel Industry Energy Consumption: predict Usage_kWh from the nine published predictors.";
    public override string TargetVariable => "Usage_kWh";
    public override string[] AllowedInputVariables =>
    [
        "Lagging_Current_Reactive_Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
        "CO2_tCO2", "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor", "NSM",
        "WeekStatus", "Day_of_week", "Load_Type"
    ];
    public override string[] VariableNames => [.. AllowedInputVariables, TargetVariable];

    public override List<List<double>> GenerateValues()
    {
        var data = VariableNames.Select(_ => new List<double>()).ToList();
        using var reader = new StreamReader(csvPath);
        using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
        csv.Read();
        csv.ReadHeader();
        while (csv.Read())
        {
            for (int column = 0; column < NumericColumns.Length; column++)
                data[column].Add(csv.GetField<double>(NumericColumns[column]));

            // GP requires numeric inputs. Preserve rows and use fixed categorical codes.
            data[6].Add(csv.GetField("WeekStatus") switch
            {
                "Weekend" => 0, "Weekday" => 1,
                var value => throw new FormatException($"Unknown WeekStatus: {value}")
            });
            data[7].Add(csv.GetField("Day_of_week") switch
            {
                "Monday" => 0, "Tuesday" => 1, "Wednesday" => 2, "Thursday" => 3,
                "Friday" => 4, "Saturday" => 5, "Sunday" => 6,
                var value => throw new FormatException($"Unknown Day_of_week: {value}")
            });
            data[8].Add(csv.GetField("Load_Type") switch
            {
                "Light_Load" => 0, "Medium_Load" => 1, "Maximum_Load" => 2,
                var value => throw new FormatException($"Unknown Load_Type: {value}")
            });
            data[9].Add(csv.GetField<double>(TargetVariable));
        }
        return data;
    }
}
