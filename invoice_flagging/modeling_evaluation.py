from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV


def train_random_forest_classifier(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 5, 6],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_classifier(model, X_test, y_test, model_name: str = "Random Forest Classifier"):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    macro_avg = report["macro avg"]
    weighted_avg = report["weighted avg"]

    print(f"\n{model_name} Performance:")
    print(f"Accuracy     : {acc:.2f}%")
    print(
        f"Macro Avg    : P={macro_avg['precision']:.2f}, "
        f"R={macro_avg['recall']:.2f}, F1={macro_avg['f1-score']:.2f}"
    )
    print(
        f"Weighted Avg : P={weighted_avg['precision']:.2f}, "
        f"R={weighted_avg['recall']:.2f}, F1={weighted_avg['f1-score']:.2f}"
    )
    print("\nFull Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    return {
        "model_name": model_name,
        "accuracy": acc,
        "macro_precision": macro_avg["precision"],
        "macro_recall": macro_avg["recall"],
        "macro_f1": macro_avg["f1-score"],
        "weighted_precision": weighted_avg["precision"],
        "weighted_recall": weighted_avg["recall"],
        "weighted_f1": weighted_avg["f1-score"],
    }