import torch.nn as nn
import torch
import os

MODEL_DIR = ''

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
            nn.Linear(self.flattened_features, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
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

if __name__ == "__main__":

    device = torch.device('cpu')
    # Use the original size for backward compatibility
    fotorrojoNet = FotorrojoNet(num_classes=2, input_size=(75, 225)).to(device)

    # Load the trained weights from checkpoint
    checkpoint_path = r"C:\git\EfficientNet-PyTorch\fotorrojoNet\results\model_best.pth.tar"

    if os.path.exists(checkpoint_path):
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
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Proceeding with random weights...")

    fotorrojoNet.eval()

    dummy_input = torch.randn(1, 3, 75, 225)

    torch.onnx.export(fotorrojoNet,
                  dummy_input,
                  MODEL_DIR + 'fotorrojoNet_75_225.onnx',
                  export_params=True,
                  input_names = ['input'],
                  output_names = ['output'],
                  opset_version=11
                 )

    if os.path.exists(MODEL_DIR + 'fotorrojoNet_75_225.onnx'):
        print('ONNX model has been saved in '+ MODEL_DIR + 'fotorrojoNet_75_225.onnx')
        if os.path.exists(checkpoint_path):
            print('ONNX model contains trained weights!')
    else:
        print('Export ONNX failed!')
