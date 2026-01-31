using Microsoft.ML;
using SentimentAnalysis.API.Models;

var mlContext = new MLContext();

// Load Data
const string dataPath = "Data/bilingual_sentiment_dataset.csv";
var dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
    dataPath, 
    hasHeader: true, 
    separatorChar: ',',
    allowQuoting: true,
    trimWhitespace: true
);
// Build Pipeline
var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(ModelInput.SentimentText))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

// Train Model
var model = pipeline.Fit(dataView);

// Save Model
var modelPath = Path.Combine(AppContext.BaseDirectory, "../../../../SentimentAnalysis.Api/MLModels/SentimentModel.zip");
mlContext.Model.Save(model, dataView.Schema, modelPath);

Console.WriteLine("Model saved successfully!");