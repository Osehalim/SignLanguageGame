import torch
from model import SignLanguageCNN

# Define the file paths
model_path = 'saved_models/sign_language_model.pth'
onnx_model_path = 'saved_models/sign_language_model.onnx'

# Load the trained PyTorch model
model = SignLanguageCNN()
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()  # Set the model to evaluation mode

# Define a dummy input that matches the model's input dimensions (1x28x28 grayscale image)
dummy_input = torch.randn(1, 1, 28, 28)

# Export the model to ONNX
torch.onnx.export(model, 
                  dummy_input, 
                  onnx_model_path, 
                  input_names=['input'], 
                  output_names=['output'], 
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                  opset_version=11)

print(f'Model has been exported to {onnx_model_path} successfully.')
