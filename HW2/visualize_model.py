import torch
from torchviz import make_dot
from models.ResNet import ResNet, BasicBlock
from models.MobileNet import MobileNetV2

def visualize(model, input_tensor, filename):
    model.eval()
    output = model(input_tensor)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(f"plots/{filename}", format="png", cleanup=True)
    print(f"Saved to plots/{filename}.png")

if __name__ == "__main__":
    dummy = torch.randn(1, 3, 32, 32)

    visualize(ResNet(BasicBlock, [2,2,2,2], num_classes=10), dummy, "resnet_architecture")
    visualize(MobileNetV2(num_classes=10), dummy, "mobilenet_architecture")