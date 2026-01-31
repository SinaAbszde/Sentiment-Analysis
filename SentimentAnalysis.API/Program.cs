using Microsoft.Extensions.ML;
using SentimentAnalysis.Api.Models;
using SentimentAnalysis.API.Models;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>().FromFile("MLModels/SentimentModel.zip");

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/predict", (PredictionEnginePool<ModelInput, ModelOutput> pool, PredictionRequest request) =>
{
    if (string.IsNullOrWhiteSpace(request.Text))
        return Results.BadRequest("Text cannot be empty");

    var input = new ModelInput { SentimentText = request.Text };

    var prediction = pool.Predict(input);
    
    return Results.Ok(new 
    { 
        IsPositive = prediction.Prediction, 
        Confidence = prediction.Probability 
    });
});

app.Run();