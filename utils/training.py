from sklearn.model_selection import GridSearchCV


def run_grid_search(estimator, param_grid, X_train, y_train, cv=5, scoring="f1_macro"):
    """Fit GridSearchCV with project-standard settings. Returns the fitted grid object."""
    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid
