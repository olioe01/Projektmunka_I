# Projektmunka I — MedPlain

Négy modell hasonlítja össze egy angol orvosi mondat besorolását: **Laikus (0)** vagy **Szakmai (1)**. A modellek a `modell osszehasonlitas (Osszegzes).ipynb` notebookból származó algoritmusok és hiperparaméterek alapján készülnek: MultinomialNB, LogisticRegression, lineáris SVC, RandomForestClassifier.

## Csapattársaknak

A repó tartalmazza a webes felületet, a Python szervert, a notebookokat, a tanítóadatokat és a négy elmentett modellt. A GitHubra feltöltés a forráskódot osztja meg; az alkalmazást mindenki a saját gépén indíthatja el.

Letöltés után Linuxon/macOS-en futtasd a projekt mappájában a `./start.sh` parancsot. Windows alatt PowerShellben:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe server.py
```

Böngészőben nyisd meg a http://localhost:8000 címet. A meglévő modellekhez nem kell újratanítás. A mentett modelleket Python 3.12 környezetben ellenőriztük; a csomagverziókat a `requirements-lock.txt` rögzíti.

## Indítás egy paranccsal

```bash
./start.sh
```

Az indító létrehozza a `.venv` környezetet, szükség esetén telepíti a függőségeket, hiányzó vagy nem betölthető modellek esetén lefuttatja a tanítást, végül elindítja az oldalt: **http://localhost:8000**. Az első indítás internetet igényel a Python-csomagok letöltéséhez. Később a modellek helyi fájlból töltődnek be; nincs tanítás egy predikciós kérés során. A futtatáshoz Python 3.10+ és a venv modul szükséges.

Manuális tanítás és indítás már telepített függőségek esetén:

```bash
.venv/bin/python train_models.py
.venv/bin/python server.py
```

## Tanítás és mentés

A `train_models.py` a `data.json` fájlból előállítja a mondatpárokat, elvégzi a notebookban szereplő üres/pontos/közel-duplikátum szűrést, és a négy osztályozót betanítja. A TF-IDF beállításai megegyeznek a notebookéval: maximum 15 000 jellemző, egy- és kétszavas kifejezések, angol stopszavak.

A kiértékelés egy tudatos ponton eltér: a tanító- és teszthalmaz cikkenként különül el, és a tesztben szereplő pontos mondatok ki vannak zárva a tanításból. Így azonos cikk szakmai és egyszerűsített mondatai nem kerülnek a két oldalra. Emiatt az újonnan mért eredmények eltérhetnek a notebookban látható számoktól. A tisztítás közel-duplikátum keresése továbbra is a teljes forrásadaton történik, ezért a teszt egyetlen, megtisztított adatkészleten végzett holdout mérés, nem független külső validáció.

Kimenetek sikeres tanítás után:

- `models/model-1.joblib` … `models/model-4.joblib`: teljes pipeline, TF-IDF átalakítóval és osztályozóval.
- `models/manifest.json`: modellnevek, fájlnevek, ténylegesen mért accuracy és F1.
- `models/training_report.json`: adatmennyiségek, csoportos felosztás és csomagverziók.

A modellfájlok **csak a tanítás sikeres lefutásával** jönnek létre. A notebook korábban rögzített kimenetei nem tartalmazzák a betanított Python-objektumokat. Modellfájlok nélkül a felület a valódi hiányzó állapotot jelzi, nem generál véletlen eredményt.

A joblib fájlok Python-kódot tölthetnek be, ezért csak saját, megbízható modellfájlokat használj. Más gépen azonos csomagverziókat használj, vagy tanítsd újra a modelleket. A reprodukcióhoz a sikeres tanítás után érdemes futtatni: `.venv/bin/python -m pip freeze > requirements-lock.txt`.

## Eredmények a felületen

A négy kártya a valódi osztálynevet és a tesztpontosságot jeleníti meg. A Naiv Bayes, logisztikus regresszió és Random Forest becsült osztályvalószínűséget is ad. A lineáris SVM a notebook beállításainak megfelelően nem használ `probability=True` opciót, ezért nem jelenítünk meg hozzá százalékot. Az osztályvalószínűség nem azonos a tesztpontossággal, és nincs külön kalibrálva. A rendszer nem diagnózist készít és nem egyszerűsíti át a szöveget.

## API

- `GET /api/models`: modellnevek, betöltéssel ellenőrzött állapot, mérési eredmények.
- `POST /api/predict`: `{"text": "English medical sentence"}`; modellenként besorolás és, ha rendelkezésre áll, osztályvalószínűség.

A szerver helyi fejlesztéshez készült, a `127.0.0.1:8000` címen figyel. Legfeljebb 10 000 karakter küldhető be. Az üres bemenet és a kizárólag ismeretlen kifejezéseket tartalmazó szöveg érthető hibaüzenetet kap. A felület a Google Fonts nélkül rendszer-betűkészlettel is használható.

## Ellenőrzés

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile server.py train_models.py
node --check web/app.js
bash -n start.sh
```

A szervertesztek a kérésellenőrzést és a predikciós adapter szerződését ellenőrzik; a valódi modellek tanítását és minőségét nem helyettesítik.
