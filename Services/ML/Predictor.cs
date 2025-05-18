using DASS.Models;
using Microsoft.ML;

namespace DASS.Services.ML;

public class Predictor
{
    private readonly string DEPRESSION_MODEL_PATH = "Services/ML/Models/onnx_depression.onnx";
    private readonly string AnxietyTransformer_PATH = "Services/ML/Models/onnx_anxiety.onnx";
    private readonly string StressTransformer_PATH = "Services/ML/Models/onnx_stress.onnx";
    private readonly PredictionEngine<DASSInput, DASSOutput> DepressionEngine;
    private readonly PredictionEngine<DASSInput, DASSOutput> AnxietyEngine;
    private readonly PredictionEngine<DASSInput, DASSOutput> StressEngine;


    public Predictor()
    {
        DepressionEngine = LoadModel(DEPRESSION_MODEL_PATH);
        AnxietyEngine = LoadModel(AnxietyTransformer_PATH);
        StressEngine = LoadModel(StressTransformer_PATH);
    }

    private static PredictionEngine<DASSInput, DASSOutput> LoadModel(string modelPath)
    {
        // Init context
        var mlContext = new MLContext();

        // get estimator and transformer
        var estimator = mlContext.Transforms.ApplyOnnxModel(
            modelFile: modelPath);
        var transformer = estimator.Fit(
            mlContext.Data.LoadFromEnumerable(new List<DASSInput>()));

        return mlContext.Model.CreatePredictionEngine<DASSInput, DASSOutput>(transformer);
    }

    public Dictionary<string, string> Predict(HomeViewModel homeViewModel)
    {
        Dictionary<string, string> prediction = [];
        var modelInput = ViewModelToInput(homeViewModel);
        DASSOutput tmpOutput;

        tmpOutput = DepressionEngine.Predict(modelInput);
        if (tmpOutput.Label != null) prediction["depression"] = tmpOutput.Label[0];

        tmpOutput = AnxietyEngine.Predict(modelInput);
        if (tmpOutput.Label != null) prediction["anxiety"] = tmpOutput.Label[0];

        tmpOutput = StressEngine.Predict(modelInput);
        if (tmpOutput.Label != null) prediction["stress"] = tmpOutput.Label[0];

        return prediction;
    }

    private static DASSInput ViewModelToInput(HomeViewModel dass42ViewModel)
    {
        var modelInput = new DASSInput();
        var responses = dass42ViewModel.Responses;

        int i = 0;
        foreach (var response in responses)
        {
            modelInput.X[i] = response.ResponseScore;
            i++;
        }

        return modelInput;
    }
}