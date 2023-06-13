from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
import subprocess
import argparse

parser = argparse.ArgumentParser('Arguments for NER inference model')

parser.add_argument('--checkpoint_path', type=str, default='./checkpoint',
                    help='path to model checkpoint files')
parser.add_argument('--onnx_path', type=str, default='./checkpoint',
                    help='path where .onnx model is saved')

args = parser.parse_args()

# Bash command for transporting the model using optimum-library
# See https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli
subprocess.run(["optimum-cli", "export", "onnx", "--model", pt_model_path, "--task", "token-classification", onnx_path], check=True, text=True)
