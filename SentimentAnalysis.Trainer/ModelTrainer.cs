using Microsoft.ML;
using SentimentAnalysis.API.Models;

namespace SentimentAnalysis.Trainer;

public class ModelTrainer
{
    private const string DataPath = "Data/sentiment_data.csv";

    public void TrainAndSave()
    {
        var mlContext = new MLContext(seed: 0);

        // Load
        IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
            DataPath, 
            hasHeader: true, 
            separatorChar: ',');

        // Pipeline
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(ModelInput.SentimentText))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

        // Train
        var model = pipeline.Fit(dataView);

        // Save
        mlContext.Model.Save(model, dataView.Schema, "../SentimentAnalysis.Api/MLModels/SentimentModel.zip");
    }
}