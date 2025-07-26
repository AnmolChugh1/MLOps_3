import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from sklearn.metrics import r2_score, mean_squared_error


class CompressedLinearModel(nn.Module):
    
    def __init__(self, input_dim, weights, bias):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.linear.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32).unsqueeze(0))
        self.linear.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return self.linear(x)


def float16_quantize(array, name="param"):
    """
    Simulate quantization using float16 precision.
    """
    array = np.array(array, dtype=np.float32)
    print(f"\n🔧 Quantizing {name} to float16...")
    quantized = array.astype(np.float16)
    restored = quantized.astype(np.float32)
    error = np.mean(np.abs(array - restored))
    print(f"   ➤ Mean Absolute Error after quantization: {error:.8f}")

    return {
        "quantized_data": quantized,
        "is_constant": False  # can be used to flag flat tensors
    }


def float16_dequantize(info):
  
    if info.get("is_constant", False):
        return np.full(len(info['quantized_data']), info['original_value'], dtype=np.float32)
    return info['quantized_data'].astype(np.float32)


def evaluate(model, X, y):
   
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        targets = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        preds = model(inputs)
        r2 = r2_score(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
    return r2, rmse


def main():
    print("🚀 Starting model quantization process...")

   
    model_sklearn = joblib.load("models/linear_model.joblib")
    weights = model_sklearn.coef_
    bias = model_sklearn.intercept_
    print(f"\n📦 Loaded model of type: {type(model_sklearn)}")
    print(f"   ➤ Coefficients shape: {weights.shape}, Bias: {bias:.4f}")

   
    quantized_weights = float16_quantize(weights, "weights")
    quantized_bias = float16_quantize([bias], "bias")

    
    dq_weights = float16_dequantize(quantized_weights)
    dq_bias = float16_dequantize(quantized_bias)[0]

    
    input_dim = len(weights)
    quantized_model = CompressedLinearModel(input_dim, dq_weights, dq_bias)

    os.makedirs("models", exist_ok=True)
    model_path = "models/quantized_pytorch_model.pt"
    torch.save(quantized_model.state_dict(), model_path)
    print(f"\n✅ Quantized model saved to: {model_path}")

   
    dummy_input = torch.tensor([[0.5] * input_dim], dtype=torch.float32)
    output = quantized_model(dummy_input)
    print(f"   ➤ Sample output: {output.item():.4f}")

   
    X_test, y_test = joblib.load("models/test_data.joblib")

    
    original_preds = model_sklearn.predict(X_test)
    r2_orig = r2_score(y_test, original_preds)
    rmse_orig = np.sqrt(mean_squared_error(y_test, original_preds))

    r2_q, rmse_q = evaluate(quantized_model, X_test, y_test)

 
    size_orig_kb = os.path.getsize("models/linear_model.joblib") / 1024
    size_q_kb = os.path.getsize(model_path) / 1024
    theory_size_orig = weights.nbytes + 4  # +4 bytes for float bias
    theory_size_q = dq_weights.nbytes + 4

    
    print("\n📊 Evaluation Results:")
    print(f"   ➤ R² (original): {r2_orig:.6f}, RMSE: {rmse_orig:.4f}")
    print(f"   ➤ R² (quantized): {r2_q:.6f}, RMSE: {rmse_q:.4f}")

    print("\n📋 FINAL COMPARISON")
    print("-" * 60)
    print(f"{'Metric':<25} {'Original':<15} {'Quantized':<15}")
    print("-" * 60)
    print(f"{'R² Score':<25} {r2_orig:<15.6f} {r2_q:<15.6f}")
    print(f"{'File Size (KB)':<25} {size_orig_kb:<15.2f} {size_q_kb:<15.2f}")
    print(f"{'Theoretical Size (bytes)':<25} {theory_size_orig:<15} {theory_size_q:<15}")
    print(f"{'Theoretical Size (KB)':<25} {theory_size_orig/1024:<15.2f} {theory_size_q/1024:<15.2f}")
    print(f"{'Compression Ratio':<25} {(size_orig_kb / size_q_kb):.2f}x")

    print("\n📌 Summary:")
    print(f"   ➤ Theoretical compression: {(theory_size_orig / theory_size_q):.2f}x")
    print(f"   ➤ Accuracy loss (R²): {r2_orig - r2_q:.6f}")

   
    joblib.dump({
        "r2_original": r2_orig,
        "r2_quantized": r2_q,
        "rmse_original": rmse_orig,
        "rmse_quantized": rmse_q,
        "size_original_kb": size_orig_kb,
        "size_quantized_kb": size_q_kb
    }, "models/comparison_results.joblib")

    print("\n✅ Quantization pipeline completed successfully!")


if __name__ == "__main__":
    main()
