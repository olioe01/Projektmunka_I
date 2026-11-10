# DL feladatok – ki mit csinál

| # | Ki | Modell | Eredményfájl |
|---|---|---|---|
| 1 | Olivér | Embedding + GlobalAveragePooling1D (baseline), közös kód, összehasonlítás | `results/baseline.json` ✅ kész |
| 2 | András | Embedding + Bidirectional(LSTM) | `results/bilstm.json` |
| 3 | Barni | Embedding + Conv1D + GlobalMaxPooling1D | `results/cnn.json` |
| 4 | Attila | Embedding + Bidirectional(GRU) | `results/bigru.json` |

## Mindenkire vonatkozik

1. `git pull`, majd `.venv/bin/python -m pip install -r requirements-lock.txt -r requirements-dl.txt` (vagy Colab: töltsd fel a `data.json`, `train_models.py`, `dl_prep.py` fájlokat).
2. Saját notebook: `dl_<szám>_<modell>.ipynb` (pl. `dl_2_bilstm.ipynb`). Minta: `dl_1_baseline.ipynb`.
3. **A `dl_prep.py`-t ne módosítsd.** Mindent onnan használj: `load_data`, `compile_model`, `fit`, `evaluate`, `plot_history`, `plot_confusion`, `save_result`.
4. Kísérletezni csak a validációs halmazon szabad (`evaluate(..., split="val")`). A tesztet egyszer, a végső modellen használd.
5. Kötelező: tanulási görbe **Dropout nélkül és Dropouttal** (`dp.plot_history({...})`), konfúziós mátrix, `model.summary()`.
6. A végén: `dp.save_result("<kulcs>", "<név>", model, history, result, seconds, data, params={...})`, majd commitold a notebookot és a `results/<kulcs>.json` fájlt.
7. 1–2 dia a saját modelledről a 2. hét végére.

## 2 – András: Bidirectional LSTM

- Modell: `Embedding(mask_zero=True) → Bidirectional(LSTM(64)) → Dropout → Dense(1, sigmoid)`
- Kísérletek: sima LSTM vs. Bidirectional; az LSTM saját `dropout` / `recurrent_dropout` paramétere a külön Dropout réteg mellett; 32 / 64 / 128 egység.
- Mérd a tanítási időt (a `dp.fit` visszaadja). Lassú, érdemes Colabon GPU-val futtatni.
- **Attilával egyeztess**: ugyanazok a beállítások (egységszám, Dropout, batch), hogy az LSTM és a GRU összevethető legyen.

## 3 – Barni: Conv1D

- Modell: `Embedding → Conv1D(128, kernel_size=5, relu) → GlobalMaxPooling1D → Dropout → Dense(1, sigmoid)`
- Kísérletek: `kernel_size` 3 / 5 / 7; a szűrők száma; ad-e valamit egy második Conv1D réteg.
- Bónusz: melyik szűrő mely n-gramokra aktiválódik a legerősebben (milyen „szakszavakat” talált a modell).

## 4 – Attila: Bidirectional GRU

- Modell: `Embedding(mask_zero=True) → Bidirectional(GRU(64)) → Dropout → Dense(1, sigmoid)`
- Kísérletek: a GRU `dropout` paramétere; egységszám; `SpatialDropout1D` az embedding után.
- Te készíted az **LSTM vs. GRU összevetést**: paraméterszám, tanítási idő, pontosság. **Andrással egyeztess** a közös beállításokról.

## Prompt az AI-nak

Másold be, a `<…>` részt cseréld a saját soraidra a fenti listából:

```
A Projektmunka_I repóban dolgozom (MedPlain: angol orvosi mondatok, szakmai=1 / laikus=0).
Olvasd el a FELADATOK.md, dl_prep.py és dl_1_baseline.ipynb fájlokat.
Feladatom: <pl. "2 – András: Bidirectional LSTM" a teljes szakasszal>.
Készíts egy dl_<szám>_<modell>.ipynb notebookot a dl_1_baseline.ipynb mintájára:
- csak a dl_prep.py függvényeit használd, a dl_prep.py-t NE módosítsd;
- a kísérleteket a validációs halmazon futtasd, 3 seed (42, 43, 44) átlagával;
- mutasd a tanulási görbét Dropout nélkül és Dropouttal;
- a végső modellt a val eredmény alapján válaszd, a tesztet csak egyszer használd;
- a végén dp.save_result("<kulcs>", ...) mentse a results/<kulcs>.json-t.
Futtasd le a notebookot, és röviden foglald össze az eredményeket magyarul.
```
