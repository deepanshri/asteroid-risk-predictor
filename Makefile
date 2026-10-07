PY ?= python

.PHONY: fetch clean train evaluate app test all

fetch:      ## check that data/raw/neo.csv exists (prints Kaggle instructions if not)
	$(PY) -m src.fetch_data

clean:      ## clean raw data -> data/processed
	$(PY) -m src.clean

train:      ## compare, tune and save the model (takes ~10 min)
	$(PY) -m src.train

evaluate:   ## figures, SHAP and error analysis -> reports/
	$(PY) -m src.evaluate

app:        ## launch the Streamlit app
	streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

all: fetch clean train evaluate test
