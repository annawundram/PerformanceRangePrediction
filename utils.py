import torch
import torch.nn.functional as F
from typing import Union

def compute_ece_for_segmentation(prob_maps, true_labels, n_bins=15):
    """
    Compute the Expected Calibration Error for a single batch in segmentation.
    Args:
        prob_maps (torch.Tensor): Probability maps from the model (shape: [batch_size, n_classes, H, W]).
        true_labels (torch.Tensor): Ground truth labels (shape: [batch_size, H, W]).
        n_bins (int): Number of bins to use for calibration error computation.

    Returns:
        float: ECE for the batch.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get mask of probabilities within this bin
        in_bin = (prob_maps > bin_lower) & (prob_maps <= bin_upper)
        prop_in_bin = in_bin.float().mean()  # Proportion of pixels in this bin

        if prop_in_bin.item() > 0:
            # Average accuracy in this bin
            accuracy_in_bin = (in_bin & (true_labels == 1)).float().sum() / in_bin.sum()

            # Average confidence in this bin
            avg_confidence_in_bin = prob_maps[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def convert_to_onehot(
    labels: torch.Tensor, num_classes: int, channel_dim: int = 1
) -> torch.Tensor:
    out = F.one_hot(labels.long(), num_classes)
    out = out.unsqueeze(channel_dim).transpose(channel_dim, out.dim())
    return out.squeeze(-1)


def find_onehot_dimension(T: torch.Tensor) -> Union[int, None]:
    if float(T.max()) <= 1 and float(T.min()) >= 0:
        for d in range(T.dim()):
            if torch.all((T.sum(dim=d) - 1) ** 2 <= 1e-5):
                return d

    # Otherwise most likely not a one-hot tensor
    return None


def harden_softmax_outputs(T: torch.Tensor, dim: int) -> torch.Tensor:
    num_classes = T.shape[dim]
    out = torch.argmax(T, dim=dim)
    return convert_to_onehot(out, num_classes=num_classes, channel_dim=dim)


if __name__ == "__main__":
    label_batch = torch.randint(0, 3, size=(32, 64, 64))
    label_onehot = convert_to_onehot(label_batch, num_classes=3)
    label_softmax = label_onehot.float() + 0.01 * torch.rand(label_onehot.shape)
    label_softmax = F.softmax(label_softmax)
    label_hard = harden_softmax_outputs(label_softmax, dim=1)

    assert torch.all(label_hard == label_onehot)
