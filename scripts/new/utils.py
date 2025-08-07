# scripts/utils.py
import os
import torch
import matplotlib.pyplot as plt

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def plot_curves(train_losses, val_losses, val_accs, out_path=None):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot([a*100 for a in val_accs], label="Val Acc (%)")
    plt.legend()
    plt.title("Validation Accuracy")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved training curves: {out_path}")
    plt.show()

# import os
# import torch
# import matplotlib.pyplot as plt

# def save_checkpoint(model, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     torch.save(model.state_dict(), path)

# def load_checkpoint(model, path, device):
#     model.load_state_dict(torch.load(path, map_location=device))
#     return model

# def plot_curves(train_losses, val_losses, val_accs, out_path=None):
#     plt.figure(figsize=(10,4))

#     plt.subplot(1,2,1)
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(val_losses, label="Val Loss")
#     plt.legend()
#     plt.title("Loss")

#     plt.subplot(1,2,2)
#     plt.plot([a*100 for a in val_accs], label="Val Acc (%)")
#     plt.legend()
#     plt.title("Validation Accuracy")

#     if out_path:
#         plt.tight_layout()
#         plt.savefig(out_path)
#         print(f"Saved training curves: {out_path}")
#     plt.show()
