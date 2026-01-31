using System.Text.Json.Serialization;
using Microsoft.ML.Data;

namespace SentimentAnalysis.API.Models;

public class ModelInput
{
    [LoadColumn(0)]
    public string SentimentText { get; set; } = string.Empty;

    [LoadColumn(1), ColumnName("Label"), JsonIgnore]
    public bool Sentiment { get; set; }
}