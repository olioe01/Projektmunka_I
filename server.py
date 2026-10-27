"""Local MedPlain web UI and four-model prediction API."""
import json
import logging
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'models' / 'manifest.json'
CACHE = {}
LOCK = RLock()
LABELS = {0: 'Laikus', 1: 'Szakmai'}


def models():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def load_model(model):
    artifact = model.get('artifact')
    if not artifact:
        raise FileNotFoundError('Nincs megadva modellfájl.')
    path = (ROOT / 'models' / artifact).resolve()
    if path.parent != (ROOT / 'models').resolve():
        raise ValueError('Érvénytelen modellútvonal.')
    stamp = path.stat().st_mtime_ns
    with LOCK:
        if model['id'] not in CACHE or CACHE[model['id']][0] != stamp:
            import joblib
            # Only load locally supplied, trusted training artifacts.
            pipeline = joblib.load(path)
            if not hasattr(pipeline, 'predict'):
                raise ValueError('A modell nem rendelkezik predict metódussal.')
            CACHE[model['id']] = (stamp, pipeline)
        return CACHE[model['id']][1]


def public_models():
    result = []
    for model in models():
        item = {key: model.get(key) for key in ('id', 'name', 'description', 'metrics')}
        try:
            load_model(model)
            item.update(ready=True, message='A modell betöltve.')
        except FileNotFoundError:
            item.update(ready=False, message='A modell még nincs betanítva. Indítsd a start.sh fájllal az alkalmazást.')
        except ImportError:
            item.update(ready=False, message='Hiányzó Python-csomag. Indítsd a start.sh fájllal az alkalmazást.')
        except Exception:
            logging.exception('Model loading failed: %s', model['id'])
            item.update(ready=False, message='A modell nem tölthető be. Újratanítás szükséges.')
        result.append(item)
    return result


def predict(text):
    results = []
    for model in models():
        try:
            pipeline = load_model(model)
            # Zero known features cannot provide a meaningful text classification.
            if pipeline.named_steps['tfidf'].transform([text]).nnz == 0:
                results.append({'id': model['id'], 'status': 'error', 'message': 'A szövegben nincs a modell által ismert kifejezés. Adj meg egy angol orvosi mondatot.'})
                continue
            label = int(pipeline.predict([text])[0])
            result = {'id': model['id'], 'status': 'ok', 'label': label, 'prediction': LABELS[label]}
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba([text])[0]
                result['probability'] = float(probabilities[list(pipeline.classes_).index(label)])
            results.append(result)
        except FileNotFoundError:
            results.append({'id': model['id'], 'status': 'unavailable', 'message': 'A betanított modell még nem érhető el. Futtasd a start.sh indítót.'})
        except ImportError:
            results.append({'id': model['id'], 'status': 'error', 'message': 'A futtatáshoz szükséges Python-csomag hiányzik. Futtasd a start.sh indítót.'})
        except Exception:
            logging.exception('Prediction failed: %s', model['id'])
            results.append({'id': model['id'], 'status': 'error', 'message': 'A modell futtatása sikertelen. A részletek a szerver naplójában találhatók.'})
    return results


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'web'), **kwargs)

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/models':
            return self.send_json(200, {'models': public_models()})
        return super().do_GET()

    def do_POST(self):
        if self.path != '/api/predict':
            return self.send_json(404, {'error': 'Ismeretlen végpont.'})
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length < 1 or length > 50000:
                return self.send_json(400, {'error': 'Érvénytelen kérésméret.'})
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                return self.send_json(400, {'error': 'JSON-objektum szükséges.'})
            text = data.get('text', '')
            if not isinstance(text, str) or not text.strip() or len(text) > 10000:
                return self.send_json(400, {'error': 'Adj meg 1–10 000 karakter hosszú szöveget.'})
        except (ValueError, TypeError, UnicodeError):
            return self.send_json(400, {'error': 'Érvénytelen JSON-kérés.'})
        return self.send_json(200, {'results': predict(text.strip())})


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('MedPlain: http://localhost:8000', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
