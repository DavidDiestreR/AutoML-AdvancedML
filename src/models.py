import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
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

        return {"alpha": new_alpha}

class KNNRegressor:
    """
    k-NN regression with internal scaling + SA neighbor proposals.

    API:
      - fit(X, y)
      - predict(X)
      - neighbour(rng) -> new instance with nearby hyperparams
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",   # "uniform" or "distance"
        p: int = 2,                 # 1 (Manhattan) or 2 (Euclidean)
        k_bounds: tuple[int, int] = (1, 100),
        use_scaler: bool = True,
    ):
        self.n_neighbors = int(n_neighbors)
        self.weights = str(weights)
        self.p = int(p)
        self.k_bounds = k_bounds
        self.use_scaler = bool(use_scaler)

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

        # clip k just in case
        k_min, k_max = self.k_bounds
        k = int(np.clip(self.n_neighbors, k_min, k_max))

        self.model_ = KNeighborsRegressor(
            n_neighbors=k,
            weights=self.weights,
            p=self.p,
            metric="minkowski",
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
        Local neighborhood moves:
          - with highest probability: k <- k +/- 1
          - sometimes: toggle weights
          - sometimes: toggle p (1 <-> 2)
        """
        k_min, k_max = self.k_bounds

        # Decide which hyperparameter to perturb (biased towards k)
        r = rng.random()

        new_k = self.n_neighbors
        new_weights = self.weights
        new_p = self.p

        if r < 0.70:
            # local move in k
            step = rng.choice([-1, 1])
            new_k = int(np.clip(new_k + step, k_min, k_max))

        elif r < 0.85:
            # toggle weights
            new_weights = "distance" if new_weights == "uniform" else "uniform"

        else:
            # toggle distance type (Manhattan vs Euclidean)
            new_p = 1 if new_p == 2 else 2

        return {
            "n_neighbors": new_k,
            "weights": new_weights,
            "p": new_p
        }


class RandomForestRegressorSA:
    """
    Random Forest regression with internal scaling + SA neighbor proposals.

    API:
      - fit(X, y)
      - predict(X)
      - neighbour(rng) -> new instance with nearby hyperparams
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: str | float = "sqrt",   # "sqrt", "log2", or float in (0,1]
        n_estimators_bounds: tuple[int, int] = (50, 500),
        max_depth_bounds: tuple[int, int] = (2, 30),
        min_samples_leaf_bounds: tuple[int, int] = (1, 20),
        use_scaler: bool = True,
        random_state: int | None = 0,
        n_jobs: int = -1,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth if (max_depth is None) else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features

        self.n_estimators_bounds = n_estimators_bounds
        self.max_depth_bounds = max_depth_bounds
        self.min_samples_leaf_bounds = min_samples_leaf_bounds

        self.use_scaler = bool(use_scaler)
        self.random_state = random_state
        self.n_jobs = n_jobs

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

        # clip ints
        ne_lo, ne_hi = self.n_estimators_bounds
        md_lo, md_hi = self.max_depth_bounds
        msl_lo, msl_hi = self.min_samples_leaf_bounds

        n_estimators = int(np.clip(self.n_estimators, ne_lo, ne_hi))
        min_samples_leaf = int(np.clip(self.min_samples_leaf, msl_lo, msl_hi))

        if self.max_depth is None:
            max_depth = None
        else:
            max_depth = int(np.clip(self.max_depth, md_lo, md_hi))

        self.model_ = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
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
        Local neighborhood moves (one per call):
          - n_estimators +/- 10
          - max_depth +/- 1 OR toggle None
          - min_samples_leaf +/- 1
          - max_features toggle among {"sqrt","log2",0.5,1.0}
        """
        ne_lo, ne_hi = self.n_estimators_bounds
        md_lo, md_hi = self.max_depth_bounds
        msl_lo, msl_hi = self.min_samples_leaf_bounds

        new_n_estimators = self.n_estimators
        new_max_depth = self.max_depth
        new_min_samples_leaf = self.min_samples_leaf
        new_max_features = self.max_features

        r = rng.random()

        if r < 0.45:
            # n_estimators move
            step = int(rng.choice([-10, 10]))
            new_n_estimators = int(np.clip(new_n_estimators + step, ne_lo, ne_hi))

        elif r < 0.75:
            # max_depth move (toggle None sometimes)
            if new_max_depth is None:
                # come back from None to a reasonable depth
                new_max_depth = int(rng.integers(md_lo, min(md_hi, 10) + 1))
            else:
                if rng.random() < 0.15:
                    new_max_depth = None
                else:
                    step = int(rng.choice([-1, 1]))
                    new_max_depth = int(np.clip(new_max_depth + step, md_lo, md_hi))

        elif r < 0.90:
            # min_samples_leaf move
            step = int(rng.choice([-1, 1]))
            new_min_samples_leaf = int(np.clip(new_min_samples_leaf + step, msl_lo, msl_hi))

        else:
            # max_features toggle (simple, robust options)
            choices = ["sqrt", "log2", 0.5, 1.0]
            # pick a different one than current
            choices = [c for c in choices if c != new_max_features]
            new_max_features = rng.choice(choices)

        return {
            "n_estimators": new_n_estimators,
            "max_depth": new_max_depth,
            "min_samples_leaf": new_min_samples_leaf,
            "max_features": new_max_features
        }

class MLP:
    """
    MLP regression with internal scaling + SA neighbor proposals.

    API:
      - fit(X, y)
      - predict(X)
      - neighbour(rng) -> new instance with nearby hyperparams
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64,),   # keep 1-2 layers
        activation: str = "relu",                      # "relu" or "tanh"
        alpha: float = 1e-4,                           # L2 regularization
        learning_rate_init: float = 1e-3,
        max_iter: int = 300,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        n_iter_no_change: int = 10,
        bounds_units: tuple[int, int] = (8, 256),
        bounds_layers: tuple[int, int] = (1, 4),
        alpha_bounds: tuple[float, float] = (1e-8, 1e1),
        lr_bounds: tuple[float, float] = (1e-5, 5e-1),
        log_step_alpha: float = 0.5,
        log_step_lr: float = 0.5,
        use_scaler: bool = True,
        random_state: int | None = 0,
    ):
        self.hidden_layer_sizes = tuple(int(x) for x in hidden_layer_sizes)
        self.activation = str(activation)
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)

        self.max_iter = int(max_iter)
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)

        self.bounds_units = bounds_units
        self.bounds_layers = bounds_layers
        self.alpha_bounds = alpha_bounds
        self.lr_bounds = lr_bounds
        self.log_step_alpha = float(log_step_alpha)
        self.log_step_lr = float(log_step_lr)

        self.use_scaler = bool(use_scaler)
        self.random_state = random_state

        self.scaler_ = None
        self.model_ = None

    def _clip_arch(self, arch: tuple[int, ...]) -> tuple[int, ...]:
        lo_u, hi_u = self.bounds_units
        lo_l, hi_l = self.bounds_layers

        # clip number of layers
        L = int(np.clip(len(arch), lo_l, hi_l))
        arch = arch[:L]

        # if too short, pad with something reasonable
        while len(arch) < L:
            arch = tuple(list(arch) + [64])

        # clip units
        arch = tuple(int(np.clip(u, lo_u, hi_u)) for u in arch)
        return arch

    def _clip_pos(self, x: float, bounds: tuple[float, float]) -> float:
        lo, hi = bounds
        return float(np.clip(x, lo, hi))

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        if self.use_scaler:
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xs = X

        arch = self._clip_arch(self.hidden_layer_sizes)
        alpha = self._clip_pos(self.alpha, self.alpha_bounds)
        lr = self._clip_pos(self.learning_rate_init, self.lr_bounds)

        self.model_ = MLPRegressor(
            hidden_layer_sizes=arch,
            activation=self.activation,
            alpha=alpha,
            learning_rate_init=lr,
            solver="adam",
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
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
        Local neighborhood moves:
          - Most often tweak architecture slightly (units +/- small step, add/remove 1 layer)
          - Sometimes perturb alpha (log space)
          - Sometimes perturb learning_rate_init (log space)
          - Rarely toggle activation
        """
        new_arch = self.hidden_layer_sizes
        new_activation = self.activation
        new_alpha = self.alpha
        new_lr = self.learning_rate_init

        r = rng.random()

        if r < 0.55:
            # tweak architecture
            arch = list(new_arch)
            lo_l, hi_l = self.bounds_layers

            # with small prob, add/remove a layer (if allowed)
            if rng.random() < 0.20 and (lo_l != hi_l):
                if len(arch) < hi_l and rng.random() < 0.5:
                    arch.append(arch[-1] if arch else 64)
                elif len(arch) > lo_l:
                    arch.pop()

            # tweak one layer width +/- step
            if len(arch) == 0:
                arch = [64]
            idx = int(rng.integers(0, len(arch)))
            step = int(rng.choice([-16, -8, 8, 16]))
            arch[idx] = arch[idx] + step

            new_arch = self._clip_arch(tuple(arch))

        elif r < 0.75:
            # alpha log move
            a = self._clip_pos(new_alpha, self.alpha_bounds)
            new_alpha = self._clip_pos(np.exp(np.log(a) + rng.normal(0.0, self.log_step_alpha)), self.alpha_bounds)

        elif r < 0.95:
            # learning rate log move
            lr = self._clip_pos(new_lr, self.lr_bounds)
            new_lr = self._clip_pos(np.exp(np.log(lr) + rng.normal(0.0, self.log_step_lr)), self.lr_bounds)

        else:
            # rare activation toggle
            new_activation = "tanh" if new_activation == "relu" else "relu"

        return {
            "hidden_layer_sizes": new_arch,
            "activation": new_activation,
            "alpha": float(new_alpha),
            "learning_rate_init": float(new_lr)
        }