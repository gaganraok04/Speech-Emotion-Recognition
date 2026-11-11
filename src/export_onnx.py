"""
Export PyTorch Model to ONNX format.

Enables deployment on:
- ONNX Runtime (C++, Python, C#, Java)
- Edge devices (Raspberry Pi, mobile)
- Client-side Web (ONNX Runtime Web / WebAssembly)
"""

from pathlib import Path
import torch
import onnx

from src.model import get_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
WEB_DIR = PROJECT_ROOT / "web"


def export_to_onnx(model_name: str = "cnn1d", in_features: int = 280, num_classes: int = 8):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"best_{model_name}.pt"
    onnx_path = MODELS_DIR / f"emotion_{model_name}.onnx"
    web_onnx_path = WEB_DIR / f"emotion_{model_name}.onnx"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = get_model(model_name=model_name, in_features=in_features, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dummy input tensor matching feature size (Batch, Features)
    dummy_input = torch.randn(1, in_features, dtype=torch.float32)

    print(f"Exporting PyTorch model to ONNX format at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["acoustic_features"],
        output_names=["logits"],
        dynamo=False,
        dynamic_axes={
            "acoustic_features": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )

    # Check validity with ONNX library
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated successfully!")

    # Copy to web directory for edge/web distribution
    import shutil
    shutil.copy2(onnx_path, web_onnx_path)
    print(f"ONNX model copied to {web_onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    export_to_onnx(model_name="cnn1d")
