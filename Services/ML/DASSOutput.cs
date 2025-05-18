using Microsoft.ML.Data;

namespace DASS.Services.ML;

public class DASSOutput
{
    [ColumnName("output_label")]
    [VectorType(1)]
    public string[]? Label { get; set; }
}