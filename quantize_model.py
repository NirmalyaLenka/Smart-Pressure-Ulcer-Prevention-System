"""
Post-Training Quantization for FPGA Deployment

Converts the trained Keras model to an 8-bit integer TFLite model
suitable for deployment on the Lattice CrossLink-NX FPGA inference engine.

Also exports weight hex files for Verilog ROM initialization.
"""

import os
import argparse
import struct
import numpy as np
import tensorflow as tf


def representative_dataset(norm_mean, norm_std, n_samples: int = 500):
    """Generate synthetic calibration data for quantization."""
    def gen():
        for _ in range(n_samples):
            # Simulate realistic sensor window
            window = np.zeros((1, 50, 4), dtype=np.float32)
            # pressure: 5-30 kPa
            window[0, :, 0] = np.random.uniform(5, 30, 50)
            # duration: 0-120 mins (increasing)
            window[0, :, 1] = np.linspace(0, 120, 50)
            # temp delta: -1 to +2 C
            window[0, :, 2] = np.random.normal(0.5, 0.5, 50)
            # spo2: 85-100 %
            window[0, :, 3] = np.random.uniform(85, 100, 50)
            # normalize
            window = (window - norm_mean) / norm_std
            yield [window]
    return gen


def quantize_model(model_path: str, output_dir: str,
                   norm_mean: np.ndarray, norm_std: np.ndarray):
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(norm_mean, norm_std)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "model_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"INT8 TFLite model saved: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")

    return tflite_model


def extract_weights_to_hex(model_path: str, output_dir: str):
    """
    Export quantized weights as hex files for Verilog $readmemh.
    Each file: one 8-bit value per line, in hex.
    """
    model = tf.keras.models.load_model(model_path)
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    layer_map = {
        "conv1": "conv1",
        "conv2": "conv2",
        "risk_score": "dense",
    }

    for layer in model.layers:
        name = layer.name
        if name not in layer_map:
            continue
        prefix = layer_map[name]
        weights = layer.get_weights()
        if not weights:
            continue

        # Kernel weights
        kernel = weights[0]
        kernel_int8 = np.clip(np.round(kernel * 128), -128, 127).astype(np.int8)
        hex_path = os.path.join(weights_dir, f"{prefix}_w.hex")
        with open(hex_path, "w") as f:
            for val in kernel_int8.flatten():
                f.write(f"{val & 0xFF:02X}\n")
        print(f"  Wrote {hex_path} ({kernel_int8.size} values)")

        # Bias
        if len(weights) > 1:
            bias = weights[1]
            bias_int8 = np.clip(np.round(bias * 128), -128, 127).astype(np.int8)
            hex_path = os.path.join(weights_dir, f"{prefix}_b.hex")
            with open(hex_path, "w") as f:
                for val in bias_int8.flatten():
                    f.write(f"{val & 0xFF:02X}\n")
            print(f"  Wrote {hex_path} ({bias_int8.size} values)")

    # Generate sigmoid LUT (256 entries)
    x = np.linspace(-8, 8, 256)
    sig = (1.0 / (1.0 + np.exp(-x)) * 255).astype(np.uint8)
    lut_path = os.path.join(weights_dir, "sigmoid_lut.hex")
    with open(lut_path, "w") as f:
        for val in sig:
            f.write(f"{val:02X}\n")
    print(f"  Wrote sigmoid LUT: {lut_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize model for FPGA")
    parser.add_argument("--model",  type=str, default="outputs/best_model.h5")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    norm_mean = np.load(os.path.join(args.output, "norm_mean.npy"))
    norm_std  = np.load(os.path.join(args.output, "norm_std.npy"))

    quantize_model(args.model, args.output, norm_mean, norm_std)
    extract_weights_to_hex(args.model, args.output)
    print("Quantization complete.")
