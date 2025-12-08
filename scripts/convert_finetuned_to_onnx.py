"""
Convert fine-tuned PyTorch embedding models to ONNX format.

Usage:
    python scripts/convert_finetuned_to_onnx.py \\
        --model-path ./my_finetuned_model.pt \\
        --output-path ./my_finetuned_model.onnx \\
        --validate
"""

import argparse
import logging
from pathlib import Path
import torch
import onnx
from onnxruntime import InferenceSession
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_finetuned_to_onnx(
    model_path: Path,
    output_path: Path,
    input_shape: tuple = (1, 512),
    opset_version: int = 14
) -> Path:
    """
    Convert fine-tuned PyTorch model to ONNX.

    Args:
        model_path: Path to PyTorch model
        output_path: Path for ONNX output
        input_shape: Input shape (batch_size, seq_length)
        opset_version: ONNX opset version

    Returns:
        Path to ONNX model
    """
    logger.info(f"Converting {model_path} to ONNX...")

    # Load PyTorch model
    try:
        model = torch.load(model_path, map_location="cpu")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    model.eval()

    # Create dummy input
    dummy_input = {
        "input_ids": torch.randint(0, 30522, input_shape, dtype=torch.long),
        "attention_mask": torch.ones(input_shape, dtype=torch.long)
    }

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )
    except Exception as e:
        logger.error(f"Failed to export to ONNX: {e}")
        raise

    logger.info(f"Model exported to {output_path}")

    return output_path


def validate_onnx_model(
    onnx_path: Path,
    pytorch_path: Path,
    test_input: dict = None
) -> bool:
    """
    Validate ONNX model against PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_path: Path to PyTorch model
        test_input: Test input (optional)

    Returns:
        True if validation passes
    """
    logger.info("Validating ONNX model...")

    # Load models
    try:
        pytorch_model = torch.load(pytorch_path, map_location="cpu")
        pytorch_model.eval()

        onnx_session = InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        logger.error(f"Failed to load models for validation: {e}")
        return False

    # Create test input
    if test_input is None:
        test_input = {
            "input_ids": torch.randint(0, 30522, (1, 128), dtype=torch.long),
            "attention_mask": torch.ones((1, 128), dtype=torch.long)
        }

    # PyTorch inference
    with torch.no_grad():
        try:
            pytorch_output = pytorch_model(**test_input)
            # Handle different output types (e.g., tuple, tensor, object with attributes)
            if hasattr(pytorch_output, "last_hidden_state"):
                pytorch_embeddings = pytorch_output.last_hidden_state.numpy()
            elif isinstance(pytorch_output, torch.Tensor):
                pytorch_embeddings = pytorch_output.numpy()
            elif isinstance(pytorch_output, tuple):
                pytorch_embeddings = pytorch_output[0].numpy()
            else:
                logger.error(f"Unknown PyTorch output type: {type(pytorch_output)}")
                return False
        except Exception as e:
            logger.error(f"PyTorch inference failed: {e}")
            return False

    # ONNX inference
    try:
        onnx_input = {
            "input_ids": test_input["input_ids"].numpy(),
            "attention_mask": test_input["attention_mask"].numpy()
        }
        onnx_output = onnx_session.run(None, onnx_input)[0]
    except Exception as e:
        logger.error(f"ONNX inference failed: {e}")
        return False

    # Compare outputs
    try:
        max_diff = np.abs(pytorch_embeddings - onnx_output).max()
        logger.info(f"Max difference: {max_diff}")

        if max_diff < 1e-4: # Slightly relaxed tolerance
            logger.info("✓ Validation passed!")
            return True
        else:
            logger.warning(f"⚠ Validation warning: max diff {max_diff} > 1e-4")
            return False
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned models to ONNX")
    parser.add_argument("--model-path", required=True, help="Path to PyTorch model")
    parser.add_argument("--output-path", required=True, help="Output path for ONNX model")
    parser.add_argument("--validate", action="store_true", help="Validate conversion")
    parser.add_argument("--opset-version", type=int, default=14, help="ONNX opset version")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Convert
        onnx_path = convert_finetuned_to_onnx(
            model_path=model_path,
            output_path=output_path,
            opset_version=args.opset_version
        )

        # Validate if requested
        if args.validate:
            if not validate_onnx_model(onnx_path, model_path):
                logger.error("Validation failed!")
                exit(1)
                
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
