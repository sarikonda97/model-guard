import subprocess
import os



blur = [1, 3, 5, 7, 9]
noise = [0.1, 0.2, 0.3, 0.4, 0.5]
budget = 50000
epochs = 50


for blur_effect in blur:
    subprocess.call(["python", "knockoff/adversary/train.py", f"models/adversary/cifar10-alexnet-adaptive-blur{blur_effect}-{budget}", "resnet34", "CIFAR10", "--budgets", f"{budget}", "-d", "0", "--pretrained", "imagenet", "--img_obfs_tcq", "blur", "--img_obfs_mag", f"{blur_effect}", "--log-interval", "100", "--epochs", f"{epochs}", "--lr", "0.01"])

for noise_effect in noise:
    subprocess.call(["python", "knockoff/adversary/train.py", f"models/adversary/cifar10-alexnet-adaptive-noise{noise_effect}-{budget}", "resnet34", "CIFAR10", "--budgets", f"{budget}", "-d", "0", "--pretrained", "imagenet", "--img_obfs_tcq", "noise", "--img_obfs_mag", f"{noise_effect}", "--log-interval", "100", "--epochs", f"{epochs}", "--lr", "0.01"])
