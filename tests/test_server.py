import io
import json
import unittest
from unittest.mock import patch
import server


class Vectorizer:
    def __init__(self, nnz=1): self.nnz = nnz
    def transform(self, texts): return self


class Classifier:
    classes_ = [0, 1]
    named_steps = {'tfidf': Vectorizer()}
    def predict(self, texts): return [1]


class ProbabilisticClassifier(Classifier):
    def predict_proba(self, texts): return [[.2, .8]]


class ServerTests(unittest.TestCase):
    def request(self, payload):
        handler = object.__new__(server.Handler)
        raw = json.dumps(payload).encode()
        handler.path = '/api/predict'
        handler.headers = {'Content-Length': str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        result = []
        handler.send_json = lambda status, data: result.append((status, data))
        handler.do_POST()
        return result[0]

    def test_reject_invalid_input(self):
        for payload in ({'text': ''}, {'text': '  '}, {'text': 3}, {'text': 'a' * 10001}, [], None):
            with self.subTest(payload=str(payload)[:30]):
                self.assertEqual(self.request(payload)[0], 400)

    def test_class_labels_and_probabilities(self):
        with patch.object(server, 'load_model', return_value=ProbabilisticClassifier()):
            status, data = self.request({'text': 'Medical text.'})
        self.assertEqual(status, 200)
        self.assertEqual(len(data['results']), 4)
        self.assertTrue(all(r['prediction'] == 'Szakmai' and r['label'] == 1 for r in data['results']))
        self.assertTrue(all(r['probability'] == .8 for r in data['results']))

    def test_svm_has_no_invented_probability(self):
        with patch.object(server, 'load_model', return_value=Classifier()):
            results = server.predict('Medical text.')
        self.assertTrue(all(r['status'] == 'ok' and 'probability' not in r for r in results))

    def test_missing_models_are_not_predictions(self):
        with patch.object(server, 'load_model', side_effect=FileNotFoundError):
            results = server.predict('Medical text.')
        self.assertTrue(all(r['status'] == 'unavailable' and 'prediction' not in r for r in results))

    def test_no_vocabulary_features(self):
        model = Classifier()
        model.named_steps = {'tfidf': Vectorizer(nnz=0)}
        with patch.object(server, 'load_model', return_value=model):
            results = server.predict('xyzzzz')
        self.assertTrue(all(r['status'] == 'error' and 'prediction' not in r for r in results))

    def test_one_model_failure_preserves_other_results(self):
        with patch.object(server, 'load_model', side_effect=[FileNotFoundError(), Classifier(), Classifier(), Classifier()]):
            results = server.predict('Medical text.')
        self.assertEqual([r['status'] for r in results], ['unavailable', 'ok', 'ok', 'ok'])

    def test_ready_requires_successful_loading(self):
        with patch.object(server, 'load_model', side_effect=ImportError):
            results = server.public_models()
        self.assertTrue(all(not r['ready'] for r in results))


if __name__ == '__main__':
    unittest.main()
