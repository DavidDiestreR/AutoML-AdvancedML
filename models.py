import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class RidgeRegressor:
    """
    Ridge regression with internal scaling + SA neighbor proposals.

    API:
      - fit(X, y)
      - predict(X)
      - neighbour(rng) -> new instance with nearby hyperparams
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        alpha_bounds: tuple[float, float] = (1e-8, 1e6),
        log_step: float = 0.35,      # smaller => more local
        use_scaler: bool = True,     # keep scaling inside the model class
        random_state: int | None = None,
    ):
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.alpha_bounds = alpha_bounds
        self.log_step = float(log_step)
        self.use_scaler = bool(use_scaler)
        self.random_state = random_state

        # learned components after fit()
        self.scaler_ = None
        self.model_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        if self.use_scaler:
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xs = X

        self.model_ = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
        )
        self.model_.fit(Xs, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return self.model_.predict(X)

    def neighbour(self, rng: np.random.Generator):
        """
        Local neighborhood move:
          alpha' = alpha * exp(N(0, log_step))
        This is 'close' in multiplicative sense (good because alpha spans orders of magnitude).
        """
        lo, hi = self.alpha_bounds
        alpha_safe = min(max(self.alpha, lo), hi)

        new_log_alpha = np.log(alpha_safe) + rng.normal(0.0, self.log_step)
        new_alpha = float(np.clip(np.exp(new_log_alpha), lo, hi))

        # Usually keep intercept fixed; uncomment if you want occasional structure moves
        new_fit_intercept = self.fit_intercept
        # if rng.random() < 0.03:
        #     new_fit_intercept = not new_fit_intercept

        return RidgeRegressor(
            alpha=new_alpha,
            fit_intercept=new_fit_intercept,
            alpha_bounds=self.alpha_bounds,
            log_step=self.log_step,
            use_scaler=self.use_scaler,
            random_state=self.random_state,
        )