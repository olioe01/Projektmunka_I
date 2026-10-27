"""Train the notebook's four classifiers; persist complete text pipelines."""
import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent

def prepare():
    rows = []
    def visit(obj, path=()):
        if not isinstance(obj, dict):
            return
        if 'abstract' in obj and 'adaptations' in obj:
            for adaptation in obj['adaptations'].values():
                for sid, text in obj['abstract'].items():
                    if sid in adaptation:
                        rows.append({'article': path[-1], 'original': text.strip(), 'simplified': adaptation[sid].strip()})
        for key, value in obj.items():
            visit(value, path + (key,))
    visit(json.loads((ROOT / 'data.json').read_text(encoding='utf-8')))
    df = pd.DataFrame(rows)
    df = df[(df.original != '') & (df.simplified != '')].drop_duplicates(['original', 'simplified']).reset_index(drop=True)
    # Same >= .85 filter as notebook; sparse blocks avoid a large dense matrix.
    matrix = TfidfVectorizer(lowercase=True, stop_words='english').fit_transform(df.original)
    remove = set()
    for start in range(0, len(df), 256):
        similarity = (matrix[start:start+256] @ matrix.T).tocoo()
        for i, j, score in zip(similarity.row, similarity.col, similarity.data):
            if start + i < j and score >= .85:
                remove.add(int(j))
    df = df.drop(index=list(remove)).reset_index(drop=True)
    classified = pd.concat([
        pd.DataFrame({'text': df.original, 'label': 1, 'article': df.article}),
        pd.DataFrame({'text': df.simplified, 'label': 0, 'article': df.article})
    ], ignore_index=True)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(classified, groups=classified.article))
    # Remove any exact sentence shared across article groups from the training set.
    test_text = set(classified.iloc[test].text.str.lower())
    train = [i for i in train if classified.iloc[i].text.lower() not in test_text]
    return classified.iloc[train], classified.iloc[test], len(df)

def main():
    train, test, pairs = prepare()
    print(f'Clean pairs: {pairs}; train: {len(train)}; test: {len(test)}', flush=True)
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), stop_words='english')
    xtrain = vectorizer.fit_transform(train.text)
    xtest = vectorizer.transform(test.text)
    specs = [
        ('model-1', 'Naiv Bayes', 'Multinomiális Naiv Bayes', MultinomialNB(alpha=1.0)),
        ('model-2', 'Logisztikus regresszió', 'Lineáris osztályozó', LogisticRegression(C=1., max_iter=1000, random_state=42)),
        ('model-3', 'Lineáris SVM', 'Tartóvektor-gép', SVC(kernel='linear', C=1., random_state=42)),
        ('model-4', 'Random Forest', '100 döntési fa', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ]
    manifest = []
    for mid, name, description, classifier in specs:
        print(f'Training {name}…', flush=True)
        classifier.fit(xtrain, train.label)
        predictions = classifier.predict(xtest)
        pipeline = Pipeline([('tfidf', vectorizer), ('classifier', classifier)])
        artifact = mid + '.joblib'
        joblib.dump(pipeline, ROOT / 'models' / artifact, compress=3)
        metrics = {'accuracy': accuracy_score(test.label, predictions), 'f1': f1_score(test.label, predictions)}
        manifest.append({'id': mid, 'name': name, 'description': description, 'artifact': artifact, 'metrics': metrics})
        print(f'Saved {artifact}: {metrics}', flush=True)
    (ROOT / 'models' / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = {'sklearn_version': sklearn.__version__, 'split': 'article-group holdout; exact test sentences excluded from training', 'random_state': 42, 'clean_pairs': pairs, 'train_samples': len(train), 'test_samples': len(test), 'train_articles': train.article.nunique(), 'test_articles': test.article.nunique(), 'models': manifest}
    (ROOT / 'models' / 'training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
