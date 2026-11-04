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
