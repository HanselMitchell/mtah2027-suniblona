# mtah2027-suniblona — ball possession from automatic tracking

More Than A Hack 2027 (Metrica Sports). Two-stage LightGBM trained on ATD-like corrupted IDSSE
(public Bundesliga tracking) + semi-Markov Viterbi decoding. Method, data, validation and
limitations: **[TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md)**.

```bash
pip install -r requirements.txt                 # Python 3.12
# point MTAH_DATA_DIR at the match bundle (folder with tactical/ and broadcast/)
python scripts/download_idsse.py                # training data (7 IDSSE matches, CC-BY 4.0)
python scripts/convert_atd.py                   # competition ATD -> data/atd/
python scripts/train_final.py                   # -> models/
python scripts/predict.py                       # -> submission/suniblona.csv
python scripts/make_pdf.py                      # -> submission/suniblona.pdf
```

The match data is not redistributable and is never committed (see `.gitignore`).
