import subprocess
import os

path = "C:/Users/jashw/Desktop/Integration/Attack Detection/src"

for i, file in enumerate(os.listdir(path + "/CIFAR-10-images/train/bird/")):
	print(f"Item {i}: ",subprocess.call(["py", "client.py", "http://localhost:8080/predict", path + "/CIFAR-10-images/train/bird/" + file]))