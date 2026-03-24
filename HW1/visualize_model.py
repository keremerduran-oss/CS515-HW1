import torch
from torchviz import make_dot
from parameters import get_params
from models.MLP import MLP


def visualize_model():
    params = get_params()
    
    model = MLP(
        input_size   = params["input_size"],
        hidden_sizes = params["hidden_sizes"],
        num_classes  = params["num_classes"],
        dropout      = params["dropout"],
        activation   = params["activation"],
        batch_norm   = params["batch_norm"],
    )

    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)

    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render("plots/model_architecture", format="pdf", cleanup=True)
    print("Model architecture saved to plots/model_architecture.pdf")


if __name__ == "__main__":
    visualize_model()