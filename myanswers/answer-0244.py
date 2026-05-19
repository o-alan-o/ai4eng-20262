import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def optimizar_svm_grid(X, y):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }

    grid_search = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=param_grid,
        cv=3,
        scoring='accuracy'
    )

    grid_search.fit(X, y)

    resultados = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_
    }

    return resultados
