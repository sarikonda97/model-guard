from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, request
from scipy import misc
import matplotlib.pyplot

import growing_set as gs
import growing_set_ops as gso
import model
import model_ops as mops

def serve_model(delta: float, oracle_path: str, model_class: model):
	print(delta, oracle_path, model_class)
	gd_agent = gs.GrowingDistanceAgent(shapiro_threshold=delta,	dist_metric=gso.l2,thr_update_rule=gso.mean_dif_std)

	allowed_extensions = ["jpg", "png", "ppm"]
	app = Flask(__name__)

	oracle = mops.load_server(oracle_path, model_class=model_class)
	oracle_predict = mops.model_handle(oracle)

	@app.route("/predict", methods=["POST"])
	def upload_image():
		if request.method == "POST":
			img_file = request.files['payload']
			if img_file and img_file.filename[-3:] in allowed_extensions:
				img_query = to_matrix(img_file)
				logits = oracle_predict(img_query)
				target_class = np.argmax(gso.softmax(logits))
				attacker_present = gd_agent.single_query(img_query, target_class)
				if attacker_present:
					print("------------Found the attacker!------------")
				res = shuffle_max_logits(logits, 3) if attacker_present else logits
				return str(res)

	app.run(port=8080, host="localhost")

def to_matrix(img_file) -> np.ndarray:
	print("The iamgfe file is:", img_file)
	return matplotlib.pyplot.imread(img_file)

def shuffle_max_logits(logits: np.ndarray, n: int) -> np.ndarray:
	logits = logits.squeeze()
	idx = logits.argsort()[-n:][::-1]
	max_elems = logits[idx]
	np.random.shuffle(max_elems)
	for i, e in zip(idx, max_elems):
		logits[i] = e
	return logits
