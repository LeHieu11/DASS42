using Microsoft.ML.Data;

namespace DASS.Services.ML;

public class DASSInput
{
    [ColumnName("X")]
    [VectorType(42)]
    public Int64[] X { get; set; } = new Int64[42];
}
