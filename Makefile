all:	g t p

g:
	python3 gen.py
t:
	python3 train.py
p:
	python3 predict.py