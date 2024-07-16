import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError

from src.data import FIVES
from src.models import UNet

import argparse
import os


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for evaluating trained models."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=dir_path,
        help="Path to directory containing the checkpoint files. The most recent checkpoint will be loaded.",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_identifier",
        type=str,
        help="Filter by checkpoint identifier such aus 'auc' or 'loss'",
        default=None,
    )

    args = parser.parse_args()

    checkpoint_dir = args.checkpoints_dir
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if args.checkpoint_identifier is not None:
        checkpoints = [f for f in checkpoints if args.checkpoint_identifier in f]

    # Sort the checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.split("=")[1].split("-")[0]))

    # Get the most recent checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

    data = "" # path to data

    test_dataset = FIVES.Fives(file_path=data, t="test")
    test_loader = DataLoader(test_dataset, batch_size=16, drop_last=False, shuffle=False)

    validation_dataset = FIVES.Fives(file_path=data, t="val", transform=None)
    validation_loader = DataLoader(
        validation_dataset, batch_size=16, drop_last=False, shuffle=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet.load_from_checkpoint(latest_checkpoint, map_location=torch.device(device))

    model.eval()  # Set the model to evaluation mode

    trainer = pl.Trainer()

    print("TESTING")
    prediction = trainer.test(model, dataloaders=test_loader)


    def calibrate_temperature(model, validation_loader, device, num_calibration_epochs):
        # Initialize the temperature scaling model

        # Use an optimizer (e.g., Adam) to find the best temperature
        optimizer = torch.optim.Adam([model.temperature_scaling.temperature], lr=0.01)

        model.to(device)

        # init ECE
        ECE = MulticlassCalibrationError(num_classes=2, n_bins=10, norm='l1')

        for epoch in range(num_calibration_epochs):
            ece = []
            for images, targets in validation_loader:
                images, targets = images.to(device), targets.to(device)

                # Forward pass through the original model
                with torch.no_grad():
                    logits = model(images)

                # Apply temperature scaling
                scaled_logits = model.temperature_scaling(logits)

                # Compute the loss (e.g., using cross-entropy for each pixel)
                loss = F.cross_entropy(scaled_logits, targets.long())

                # log ECE
                ece.append(ECE(scaled_logits, targets.long()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(
                f"-- EPOCH {epoch}; Temperature={model.temperature_scaling.temperature.item()}; ECE={torch.mean(torch.tensor(ece))}"
            )

    print("CALIBRATING")
    print(f"Using device: {device}")
    calibrate_temperature(
        model, validation_loader, device=device, num_calibration_epochs=10
    )

    print("TESTING AGAIN")
    prediction = trainer.test(model, dataloaders=test_loader)

    print("Temperature: ", model.temperature_scaling.temperature.item())
