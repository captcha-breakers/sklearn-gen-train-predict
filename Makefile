all:	train predict

train:
	python3 main.py
predict:
	python3 predictor.py