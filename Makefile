PY:= python3

new:
	$(PY) new-char-gen.py
char:
	$(PY) char-gen.py
cap:
	$(PY) captcha-gen.py
t:
	$(PY) train.py
p:
	$(PY) predict.py