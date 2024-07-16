import gc

from src.models import PHISeg, ProbUNet, UNet, UNetMCDropout
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from src.data import FIVES
import pandas as pd

def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice coefficient for a given pair of ground truth and predicted segmentation masks. Mean over labels.

    Parameters:
    - y_true (numpy.ndarray): Ground truth segmentation mask.
    - y_pred (numpy.ndarray): Predicted segmentation mask.

    Returns:
    - float: The Dice coefficient.
    """
    dice = 0
    for value in [0, 1]:
        true_binary = y_true == value
        pred_binary = y_pred == value
        if true_binary.sum() + pred_binary.sum() == 0: return dice
        intersection = np.logical_and(true_binary, pred_binary)
        dice += (2 * intersection.sum()) / (true_binary.sum() + pred_binary.sum())
    return dice / 2


def predict_dice(mat):
    """
        Compute the predicted Dice coefficient for a given predicted segmentation mask by calculating a confusion
        matrix.

        Parameters:
        - mat matrix of predicted segmentation

        Returns:
        - float: The Dice coefficient.
        """
    positives = mat > 0.5
    negatives = mat <= 0.5

    TP = mat[positives].sum()
    FP = positives.sum() - TP
    FN = mat[negatives].sum()
    # TN = negatives.sum() - FN

    return (2 * TP) / (2 * TP + FP + FN)


def get_samples(model, input, n_samples, temp):
    if model.__class__.__name__ == "UNet":
       model_pred = model.predict(input)

       # temperature scale
       model.temperature_scaling.temperature = torch.nn.Parameter(torch.ones(1) * temp)
       model_pred = model.temperature_scaling(model_pred)

       s = torch.nn.Softmax(dim=1)
       samples_s = s(model_pred)  # 1x2x320x320
       samples_s = samples_s[0].detach().cpu().numpy()  # 2x320x320
       samples_s = samples_s[1]  # 320x320 only use softmax for label 1

       samples = torch.argmax(model_pred, dim=1)  # 1x320x320

       samples = samples.detach().cpu().numpy()
       samples = samples[0]  # remove batch dimension -> 320x320
       samples = samples.astype("uint8")

       return samples, samples_s

    else:
        model_pred = model.predict_output_samples(input, N=n_samples)  # 1xnx2x320x320

        s = torch.nn.Softmax(dim=2)
        samples_s = s(model_pred)  # 1x100x2x320x320
        samples_s = samples_s[0].detach().cpu().numpy()  # nx2x320x320
        samples_s = samples_s[:, 1, :, :]  # nx320x320 only use softmax for label 1

        samples = torch.argmax(model_pred, dim=2)  # 1xnx320x320

        samples = samples.detach().cpu().numpy()
        samples = samples[0]  # remove batch dimension -> nx320x320
        samples = samples.astype("uint8")

        return samples, samples_s


def predict(image, n_samples, model, temp):
    """
        Predict n samples for a given sample

        Parameters:
        - image Image for which to predict
        - n_samples Number of samples to draw from model
        - model Model that predicts image
        - temp Actual temp. scaling

        Returns:
        - samples hardened samples in shape n_samples x 320 x 320
        - samples_s softmax samples in shape n_samples x 2 x 320 x 320
        """
    image = torch.unsqueeze(image, 0)

    return get_samples(model, image, n_samples, temp)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for evaluating trained models."
    )
    parser.add_argument(
        "--checkpoints_dir",
        help="Path to directory containing the checkpoint files. The most recent checkpoint will be loaded.",
        nargs='+',
        required=True,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="Number of samples to be drawn",
        default=100
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Either 'cpu' or 'gpu'",
        default='cpu',
        required=True
    )

    args = parser.parse_args()
    checkpoint_dir = args.checkpoints_dir
    device = args.device

    # data
    data_path = "/FIVES.h5"
    cal_dataset = FIVES.Fives(file_path=data_path, t="cal", transform=None)
    test_dataset = FIVES.Fives(file_path=data_path, t="test", transform=None)

    # model
    if "phiseg" in checkpoint_dir[0]:
        model = PHISeg.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
        model.eval()
        saving_name = "phisegFIVES"
        temp = 1
    elif "probunet" in checkpoint_dir[0]:
        model = ProbUNet.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
        model.eval()
        saving_name = "probunetFIVES"
        temp = 1
    elif "mcdropout" in checkpoint_dir[0]:
        model = UNetMCDropout.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
        model.eval()
        saving_name = "mcdropoutFIVES"
        temp = 1
    else:
        model1 = UNet.load_from_checkpoint(checkpoint_dir[0], map_location=torch.device(device))
        model2 = UNet.load_from_checkpoint(checkpoint_dir[1], map_location=torch.device(device))
        model3 = UNet.load_from_checkpoint(checkpoint_dir[2], map_location=torch.device(device))
        model4 = UNet.load_from_checkpoint(checkpoint_dir[3], map_location=torch.device(device))
        model5 = UNet.load_from_checkpoint(checkpoint_dir[4], map_location=torch.device(device))
        model6 = UNet.load_from_checkpoint(checkpoint_dir[5], map_location=torch.device(device))
        model7 = UNet.load_from_checkpoint(checkpoint_dir[6], map_location=torch.device(device))
        model8 = UNet.load_from_checkpoint(checkpoint_dir[7], map_location=torch.device(device))
        model9 = UNet.load_from_checkpoint(checkpoint_dir[8], map_location=torch.device(device))
        model10 = UNet.load_from_checkpoint(checkpoint_dir[9], map_location=torch.device(device))

        model1.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        model5.eval()
        model6.eval()
        model7.eval()
        model8.eval()
        model9.eval()
        model10.eval()

        saving_name = "unetFIVES"
        temps = [0.779, 0.848, 1.041, 0.713, 0.923, 0.777, 0.778, 0.767, 0.924, 0.846]  # for FIVES ensemble obtained using calibrate_unet.py

    n_samples = args.n_samples  # number of samples from model
    n = len(cal_dataset)  # number of images in calibration set
    alpha = 0.1  # conformal prediction

    # ------------- calculate on calibration set -------------
    gt_dice = []
    pred_dice = []
    pred_var = []
    for i in range(n):
        image, label = cal_dataset[i]

        if len(checkpoint_dir) > 1:  # ensemble
            ensemble = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
            samples = []
            samples_s = []
            for temp, m in zip(temps, ensemble):
                sample, sample_s = predict(image, n_samples, m, temp)
                samples.append(sample)
                samples_s.append(sample_s)
            samples = np.asarray(samples)
            samples_s = np.asarray(samples_s)
        else:
            samples, samples_s = predict(image, n_samples, model, temp)

        # calculate GT dice
        result = np.asarray(list(map(lambda x: dice_coefficient(label, x), samples)))
        gt_dice.append(np.mean(result))

        # calculate predicted dice
        result = np.asarray(list(map(predict_dice, samples_s)))
        pred_dice.append(np.mean(result))
        pred_var.append(np.var(result))

        del image, samples, samples_s
        gc.collect()

    # ------------- get score function and qhat -------------
    scores = list(map(lambda y, f, u: abs((y - f)) / u, gt_dice, pred_dice, pred_var))
    qhat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n)

    # ------------- apply to test set to get predicted performance range -------------
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]

        if len(checkpoint_dir) > 1:
            ensemble = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
            samples = []
            samples_s = []
            for temp, m in zip(temps, ensemble):
                sample, sample_s = predict(image, n_samples, m, temp)
                samples.append(sample)
                samples_s.append(sample_s)
            samples = np.asarray(samples)
            samples_s = np.asarray(samples_s)
        else:
            samples, samples_s = predict(image, n_samples, model, temp)

        # calculate GT dice
        result = np.asarray(list(map(lambda x: dice_coefficient(label, x), samples)))
        gt_dice.append(np.mean(result))

        # calculate predicted dice
        result = np.asarray(list(map(predict_dice, samples_s)))
        pred_dice.append(np.mean(result))
        pred_var.append(np.var(result))

        del image, samples, samples_s
        gc.collect()

    gt_dice = np.asarray(gt_dice)
    pred_var = np.asarray(pred_var)
    pred_dice = np.asarray(pred_dice)

    # save stuff
    np.savez("performanceprediction" + saving_name, gt_dice=gt_dice, pred_dice=pred_dice, pred_var=pred_var, qhat=qhat)

    # ------------- data -------------
    # use only test set results; all values starting from n (which is size of cal set)
    gt_dice = gt_dice[n:]
    pred_dice = pred_dice[n:]
    pred_var = pred_var[n:]

    sorted_gt_dice = np.sort(gt_dice)
    ind_sorted_gt_dice = np.argsort(gt_dice)
    sorted_pred_dice = pred_dice[ind_sorted_gt_dice]
    sorted_pred_var = pred_var[ind_sorted_gt_dice]

    lower = sorted_pred_dice - qhat * sorted_pred_var
    upper = np.asarray(sorted_pred_dice + qhat * sorted_pred_var)
    upper[upper > 1] = 1
    lower[lower < 0] = 0
    x_values = np.asarray(range(len(sorted_gt_dice)))

    # label quality; quality = 1 for good quality; quality = 0 for bad quality
    # (>=2 of 3 quality features are marked as 0)
    quality_file = pd.read_excel("FIVES/Quality Assessment.xlsx", sheet_name="Test")
    # Count the number of 0's in each row for the specified columns
    count_zeros = (quality_file[['IC', 'Blur', 'LC']] == 0).sum(axis=1)

    # Set 'quality' to 0 if at least two elements are 0, otherwise set to 1
    quality_file["quality"] = (count_zeros < 2).astype(int)

    # sort quality labels in dataframe by indices in H5 file
    quality_sorted = np.zeros((160))
    for i in range(160):
        byte_str = test_dataset.get_id(i)

        decoded_str = byte_str.decode('utf-8')

        # extract the numeric part from the string
        numeric_part = ''.join(filter(str.isdigit, decoded_str))

        # convert the numeric part to an integer
        result = int(numeric_part)

        quality_sorted[i] = quality_file["quality"][result - 1]

    quality_sorted = quality_sorted[ind_sorted_gt_dice]
    # Create masks for marker 0 and marker 1
    mask_0 = quality_sorted == 0
    mask_1 = quality_sorted == 1

    # Apply masks to x_values and sorted_gt_dice
    x_values_1 = x_values[mask_1]
    x_values_0 = x_values[mask_0]
    sorted_gt_dice_1 = sorted_gt_dice[mask_1]
    sorted_gt_dice_0 = sorted_gt_dice[mask_0]

    # ------------- plot overview -------------
    fig, ax = plt.subplots()
    ax.fill_between(x_values, lower, upper,
                    color='gray', alpha=0.3)
    ax.scatter(x_values, sorted_pred_dice, label='Estimated DSC', s=2)
    ax.scatter(x_values_0, sorted_gt_dice_0, color='red', label='True DSC - bad quality', s=2, marker="o")
    ax.scatter(x_values_1, sorted_gt_dice_1, color='green', label='True DSC - good quality', s=2, marker="X")
    # Add legend
    plt.legend(loc='lower right')

    # Add labels
    plt.xlabel('image index')
    plt.ylabel('DSC')


    # -------------- size stratified plot ---------------
    def interval_percentage(lower_bounds, upper_bounds, ys):
        count_within_interval = 0
        for lower, upper, y in zip(lower_bounds, upper_bounds, ys):
            if lower <= y <= upper:
                count_within_interval += 1

        percentage = (count_within_interval / len(ys)) * 100
        return percentage

    # ------------- print numerical results -------------
    print("Percentage of gt dice scores in intervall: ",
          interval_percentage(pred_dice - qhat * pred_var, pred_dice + qhat * pred_var, gt_dice))
    print("Mean width of bound: ", 2 * qhat * np.mean(pred_var))

    MSE = np.square(np.subtract(gt_dice, pred_dice)).mean()
    print("MSE: ", MSE)