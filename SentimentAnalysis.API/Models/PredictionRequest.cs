namespace SentimentAnalysis.Api.Models;

public record PredictionRequest
{
    public string Text { get; init; } = string.Empty;
}