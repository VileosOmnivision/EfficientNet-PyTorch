import torch.nn as nn
import torch
import os


class FotorrojoNet(nn.Module):
    def __init__(self, num_classes=2, input_size=(75, 225)):
        super(FotorrojoNet, self).__init__()
        self.input_size = input_size

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate flattened features size dynamically
        self.flattened_features = self._get_flattened_size()

        self.fc_block = nn.Sequential(
            nn.Linear(self.flattened_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def _get_flattened_size(self):
        """Calculate the size after conv layers for any input size"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.input_size)
            x = self.conv_block1(dummy_input)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            return x.numel()  # Total number of elements

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.fc_block(x)
        return x

def export_onnx_model(checkpoint_path=None, output_dir='', session_name='', num_classes=2, input_size=(75, 225)):
    """
    Export a trained FotorrojoNet model to ONNX format.

    Args:
        checkpoint_path (str): Path to the trained model checkpoint
        output_dir (str): Directory where to save the ONNX file
        num_classes (int): Number of output classes
        input_size (tuple): Input image dimensions (H, W)

    Returns:
        bool: True if export was successful, False otherwise
    """
    device = torch.device('cpu')

    # Create model instance
    fotorrojoNet = FotorrojoNet(num_classes=num_classes, input_size=input_size).to(device)

    # Load trained weights if checkpoint path is provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load the state dict (handle DataParallel wrapper if present)
        state_dict = checkpoint['state_dict']

        # Remove 'module.' prefix if it exists (from DataParallel/DistributedDataParallel)
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

        fotorrojoNet.load_state_dict(state_dict)
        print(f"Successfully loaded weights from epoch {checkpoint['epoch']}")
        print(f"Best accuracy: {checkpoint['best_acc1']:.2f}%")
        weights_loaded = True
    else:
        if checkpoint_path:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Proceeding with random weights...")
        weights_loaded = False

    fotorrojoNet.eval()

    # Create dummy input for export
    dummy_input = torch.randn(1, 3, *input_size)

    # Generate ONNX filename
    onnx_filename = f'fotorrojoNet_{session_name}_{input_size[0]}x{input_size[1]}.onnx'
    onnx_path = os.path.join(output_dir, onnx_filename)

    try:
        # Export to ONNX
        torch.onnx.export(fotorrojoNet,
                      dummy_input,
                      onnx_path,
                      export_params=True,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11
                     )

        if os.path.exists(onnx_path):
            print(f'ONNX model has been saved at: {onnx_path}')
            if weights_loaded:
                print('ONNX model contains trained weights!')
            return True
        else:
            print('Export ONNX failed!')
            return False

    except Exception as e:
        print(f'ONNX export failed with error: {e}')
        return False

if __name__ == "__main__":
    # Default export for standalone usage
    checkpoint_path = r"C:\git\EfficientNet-PyTorch\fotorrojoNet\training_history\results\model_best.pth.tar"
    output_path = r"C:\git\EfficientNet-PyTorch\fotorrojoNet\training_history\results"
    export_onnx_model(checkpoint_path=checkpoint_path, output_dir=output_path)
