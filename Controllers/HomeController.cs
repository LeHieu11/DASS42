using System.Diagnostics;
using Microsoft.AspNetCore.Mvc;
using DASS.Models;
using DASS.Const;
using DASS.Services.ML;

namespace DASS.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;
    private readonly Predictor _predictor;

    public HomeController(ILogger<HomeController> logger, Predictor predictor)
    {
        _logger = logger;
        _predictor = predictor;
    }

    public IActionResult Index()
    {
        var viewModel = new HomeViewModel
        {
            Responses = [.. Dass42Questions.Questions
            .Select((q) => new Dass42Response
            {
                QuestionText = q
            })]
        };

        return View(viewModel);
    }

    public IActionResult Result(ResultViewModel resultViewModel)
    {
        return View(resultViewModel);
    }

    [HttpPost]
    public IActionResult Submit(HomeViewModel model)
    {
        if (!ModelState.IsValid) return RedirectToAction("Error");

        var prediction = _predictor.Predict(model);

        var resultViewModel = new ResultViewModel()
        {
            DepressionLevel = prediction["depression"] ?? "Unknown",
            AnxietyLevel = prediction["anxiety"] ?? "Unknown",
            StressLevel = prediction["stress"] ?? "Unknown"
        };

        return RedirectToAction("Result", resultViewModel);
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
