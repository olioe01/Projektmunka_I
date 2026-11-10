# MedPlain – Projektmunka I

Angol orvosi mondatok besorolása laikus vagy szakmai kategóriába négy gépi tanulási modellel.

## Telepítés és indítás

Szükséges: Git, Python 3.14 és internetkapcsolat az első telepítéshez. A betanított modellek a repóban vannak, külön tanítás nem szükséges.

```bash
git clone https://github.com/olioe01/Projektmunka_I.git
cd Projektmunka_I
```

### Linux / macOS

```bash
./start.sh
```

Az indító létrehozza a virtuális környezetet, telepíti a függőségeket és elindítja a szervert. Ha a modellek hiányoznak vagy nem tölthetők be, előbb újratanítja őket.

Ha Ubuntu/Debian alatt a virtuális környezet létrehozásakor az `ensurepip` hiányzik, telepítsd a Python verziódhoz tartozó venv csomagot, majd futtasd újra az indítót:

```bash
sudo apt install python3.14-venv
```

### Windows (PowerShell)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe server.py
```

## Megnyitás és leállítás

A böngészőben nyisd meg: **http://localhost:8000**

A szerver leállítása a terminálban: `Ctrl+C`.

Következő indításkor Linuxon/macOS-en ismét a `./start.sh`, Windowson a `.\.venv\Scripts\python.exe server.py` parancsot futtasd a projekt mappájában.

## Mélytanulásos modellek (DL)

**Ki mit csinál, és mit írj az AI-nak: [FELADATOK.md](FELADATOK.md)**

Közös keret a négy DL modellhez: `dl_prep.py`. Ugyanazt a tisztítást (7432 pár) és a `train_models.py` cikkenkénti felosztását használja, mint a klasszikus modellek. Ehhez jön még egy validációs halmaz (a train 12%-a cikkenként), egyetlen, csak train-en illesztett tokenizer (`num_words=20 000`, `maxlen` = 95. percentilis = 48), valamint közös tanítási és kiértékelési függvények.

```bash
.venv/bin/python -m pip install -r requirements-lock.txt -r requirements-dl.txt
.venv/bin/python dl_prep.py        # ellenőrzés: felosztás, maxlen, teszt-ujjlenyomat
```

Colabon: töltsd fel a `data.json`, `train_models.py` és `dl_prep.py` fájlokat (vagy klónozd a repót), majd `import dl_prep as dp`.

Saját modell sablon:

```python
import dl_prep as dp
from tensorflow import keras
from tensorflow.keras import layers

dp.set_seeds()
data = dp.load_data()
model = keras.Sequential([
    keras.Input(shape=(data.maxlen,)),
    layers.Embedding(data.vocab_size, dp.EMBED_DIM, mask_zero=True),
    # ... saját rétegek ...
    layers.Dense(1, activation="sigmoid"),
])
dp.compile_model(model)                      # Adam + binary_crossentropy
history, seconds = dp.fit(model, data)       # EarlyStopping(val_loss, patience=3, restore_best_weights)
dp.evaluate(model, data, split="val")        # kísérletezéshez CSAK a val halmaz
dp.plot_history({"Dropout nélkül": h0, "Dropouttal": h1})

result = dp.evaluate(model, data, split="test")   # a legvégén, egyszer
dp.save_result("bilstm", "Embedding + BiLSTM", model, history, result, seconds, data, params={...})
```

A `save_result` egy `results/<kulcs>.json` fájlt ír (kulcsok: `baseline`, `bilstm`, `cnn`, `bigru`). Ezt commitold, és a `dl_osszehasonlitas.ipynb` automatikusan beveszi a közös táblába és ábrákba, a 4 klasszikus modell mellé.

| Notebook | Tartalom |
|---|---|
| `dl_1_baseline.ipynb` | 1. Embedding + GlobalAveragePooling1D + Dense: `mask_zero` ki/be, Dropout-arányok, tanulási görbék |
| `dl_osszehasonlitas.ipynb` | 4 klasszikus + DL modellek: tábla, oszlopdiagram, konfúziós mátrixok, tanulási görbék |
