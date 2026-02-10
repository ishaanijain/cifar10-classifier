import torch
import numpy as np
import onnx
import onnxruntime as ort

from config import DEVICE, BEST_MODEL_PATH, ONNX_MODEL_PATH, CLASS_NAMES, CIFAR10_MEAN, CIFAR10_STD
from model import build_resnet18


def export():
    model = build_resnet18()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)

    torch.onnx.export(
        model, dummy_input, ONNX_MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13,
    )
    print(f"Exported ONNX model to {ONNX_MODEL_PATH}")

    # quick sanity check
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")


def test_inference():
    """Run a single random image through the onnx model just to verify it works."""
    session = ort.InferenceSession(ONNX_MODEL_PATH)

    # fake a normalized cifar-10 image
    img = np.random.randn(1, 3, 32, 32).astype(np.float32)
    outputs = session.run(None, {"input": img})
    logits = outputs[0]

    pred = np.argmax(logits, axis=1)[0]
    print(f"\nTest inference on random noise:")
    print(f"  Predicted class: {CLASS_NAMES[pred]} (index {pred})")
    print(f"  Logits: {logits[0][:5]}... (showing first 5)")
    print("\nONNX inference is working.")


if __name__ == "__main__":
    export()
    test_inference()
