PY:= python3

char:
	$(PY) char-gen.py
cap:
	$(PY) captcha-gen.py
t:
	$(PY) train.py
p:
	$(PY) predict.py