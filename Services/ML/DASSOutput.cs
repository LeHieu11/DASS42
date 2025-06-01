using Microsoft.ML.Data;

namespace DASS.Services.ML;

public class DASSOutput
{
    [ColumnName("label")]
    [VectorType(1)]
    public Int64[] Label { get; set; } = new Int64[1];
}