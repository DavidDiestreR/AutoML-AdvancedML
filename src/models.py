import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

class PolynomialRegressorSA:
    """
    Polynomial regression with optional regularization and SA neighbor proposals.

    API:
      - fit(X, y)
      - predict(X)
      - neighbour(rng) -> dict with nearby hyperparams
    """

    def __init__(
        self,
        max_degree: int = 3,
        degree_mask: tuple[int, ...] | None = None,  # bit i => use degree i+1 terms
        regularization: str = "ridge",               # "none", "ridge", "lasso"
        alpha: float = 1.0,
        fit_intercept: bool = True,
        degree_bounds: tuple[int, int] = (1, 6),
        alpha_bounds: tuple[float, float] = (1e-8, 1e6),
        log_step: float = 0.35,      # smaller => more local
        large_jump_prob: float = 0.20,
        large_log_step: float = 1.30,
        lasso_max_iter: int = 3000,
        lasso_tol: float = 1e-4,
        use_scaler: bool = True,     # keep scaling inside the model class
        random_state: int | None = None,
    ):
        self.max_degree = int(max_degree)
        self.degree_mask = tuple(int(x) for x in degree_mask) if degree_mask is not None else None
        self.regularization = str(regularization).lower()
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.degree_bounds = degree_bounds
        self.alpha_bounds = alpha_bounds
        self.log_step = float(log_step)
        self.large_jump_prob = float(large_jump_prob)
        self.large_log_step = float(large_log_step)
        self.lasso_max_iter = int(lasso_max_iter)
        self.lasso_tol = float(lasso_tol)
        self.use_scaler = bool(use_scaler)
        self.random_state = random_state

        # learned components after fit()
        self.scaler_ = None
        self.poly_ = None
        self.active_columns_ = None
        self.effective_degree_mask_ = None
        self.model_ = None

    @staticmethod
    def _clip_alpha(alpha: float, bounds: tuple[float, float]) -> float:
        lo, hi = bounds
        return float(np.clip(alpha, lo, hi))

    def _normalize_degree_mask(self, n_degrees: int) -> tuple[int, ...]:
        if n_degrees < 1:
            raise ValueError("n_degrees must be >= 1.")

        if self.degree_mask is None:
            mask = [1] * n_degrees
        else:
            mask = [1 if int(v) != 0 else 0 for v in self.degree_mask]
            if len(mask) < n_degrees:
                mask = mask + [1] * (n_degrees - len(mask))
            else:
                mask = mask[:n_degrees]

        if sum(mask) == 0:
            mask[0] = 1

        return tuple(mask)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)

        if self.use_scaler:
            self.scaler_ = StandardScaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xs = X

        deg_lo, deg_hi = self.degree_bounds
        effective_degree = int(np.clip(self.max_degree, deg_lo, deg_hi))
        self.effective_degree_mask_ = self._normalize_degree_mask(effective_degree)

        self.poly_ = PolynomialFeatures(
            degree=effective_degree,
            include_bias=False,
        )
        X_poly = self.poly_.fit_transform(Xs)
        term_degrees = self.poly_.powers_.sum(axis=1)

        active_degrees = {
            deg for deg, bit in enumerate(self.effective_degree_mask_, start=1) if bit == 1
        }
        self.active_columns_ = np.isin(term_degrees, list(active_degrees))
        X_selected = X_poly[:, self.active_columns_]

        reg = self.regularization
        alpha = self._clip_alpha(self.alpha, self.alpha_bounds)
        if reg == "none":
            self.model_ = LinearRegression(fit_intercept=self.fit_intercept)
        elif reg == "ridge":
            self.model_ = Ridge(
                alpha=alpha,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state,
            )
        elif reg == "lasso":
            self.model_ = Lasso(
                alpha=alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.lasso_max_iter,
                tol=self.lasso_tol,
                random_state=self.random_state,
            )
        else:
            raise ValueError("regularization must be one of: 'none', 'ridge', 'lasso'.")

        self.model_.fit(X_selected, y)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.asarray(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        X_poly = self.poly_.transform(X)
        X_selected = X_poly[:, self.active_columns_]
        return self.model_.predict(X_selected)

    def neighbour(self, rng: np.random.Generator):
        """
        Local neighborhood moves:
          - Most often: flip one degree bit (0 <-> 1)
          - Sometimes: switch regularization among {"none","ridge","lasso"}
          - Sometimes: alpha log move
          - Rarely: move max_degree by +/-1
        """
        deg_lo, deg_hi = self.degree_bounds
        current_degree = int(np.clip(self.max_degree, deg_lo, deg_hi))
        mask = list(self._normalize_degree_mask(current_degree))

        new_degree = current_degree
        new_mask = mask.copy()
        new_regularization = self.regularization
        new_alpha = self.alpha

        r = rng.random()
        if r < 0.60:
            idx = int(rng.integers(0, len(new_mask)))
            new_mask[idx] = 1 - new_mask[idx]
            if sum(new_mask) == 0:
                new_mask[idx] = 1

        elif r < 0.75:
            choices = ["none", "ridge", "lasso"]
            choices = [c for c in choices if c != new_regularization]
            new_regularization = choices[int(rng.integers(0, len(choices)))]

        elif r < 0.90:
            lo, hi = self.alpha_bounds
            alpha_safe = min(max(self.alpha, lo), hi)
            step_scale = self.large_log_step if rng.random() < self.large_jump_prob else self.log_step
            new_log_alpha = np.log(alpha_safe) + rng.normal(0.0, step_scale)
            new_alpha = float(np.clip(np.exp(new_log_alpha), lo, hi))

        else:
            step = -1 if rng.random() < 0.5 else 1
            new_degree = int(np.clip(current_degree + step, deg_lo, deg_hi))
            if new_degree > len(new_mask):
                new_mask = new_mask + [1] * (new_degree - len(new_mask))
            elif new_degree < len(new_mask):
                new_mask = new_mask[:new_degree]
                if sum(new_mask) == 0:
                    new_mask[0] = 1

        return {
            "max_degree": int(new_degree),
            "degree_mask": tuple(int(v) for v in new_mask),
            "regularization": new_regularization,
            "alpha": float(new_alpha),
        }

class KNNRegressor:
    """
    k-NN regression with internal scaling + SA neighbor proposals.

    API:
      - fit(X, y)
      - predict(X)
      - neighbour(rng) -> dict with nearby hyperparams
    """

    def __init__(
        self,
        n_neighbors: int | None = None,
        weights: str = "uniform",   # "uniform" or "distance"
        p: int = 2,                 # 1 (Manhattan) or 2 (Euclidean)
        k_bounds: tuple[int, int] = (1, 100),
        use_scaler: bool = True,
    ):
        self.n_neighbors = None if n_neighbors is None else int(n_neighbors)
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

        # k upper bound is always the training subset size (CV-fold aware).
        k_min, _ = self.k_bounds
        n_samples = int(Xs.shape[0])
        dynamic_k_max = max(k_min, n_samples)
        if self.n_neighbors is None:
            # Auto-init k at 2% of available samples.
            base_k = int(round(0.02 * n_samples))
        else:
            base_k = int(self.n_neighbors)
        k = int(np.clip(base_k, k_min, dynamic_k_max))

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
          - with highest probability: k move with step size proportional to data size
          - sometimes: toggle weights
          - sometimes: toggle p (1 <-> 2)
        """
        k_min, k_max = self.k_bounds
        k_scale = int(max(k_min, k_max))

        # Decide which hyperparameter to perturb (biased towards k)
        r = rng.random()

        new_k = self.n_neighbors
        new_weights = self.weights
        new_p = self.p

        if new_k is None:
            new_k = max(k_min, int(round(0.02 * k_scale)))

        if r < 0.70:
            # proportional move in k based on the expected data size scale
            base_step = max(1, int(round(0.01 * k_scale)))
            step_options = [base_step, 2 * base_step, 5 * base_step, 10 * base_step]
            magnitude = int(rng.choice(step_options, p=[0.45, 0.30, 0.20, 0.05]))
            step = magnitude if rng.random() < 0.5 else -magnitude
            candidate_k = int(max(k_min, new_k + step))
            # Avoid null proposals at lower bound when possible.
            if candidate_k == new_k and new_k > k_min:
                candidate_k = int(max(k_min, new_k - step))
            new_k = candidate_k

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
      - neighbour(rng) -> dict with nearby hyperparams
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 15,
        min_samples_leaf: int = 2,
        max_features: str | float = "sqrt",   # "sqrt", "log2", or float in (0,1]
        n_estimators_bounds: tuple[int, int] = (50, 500),
        max_depth_bounds: tuple[int, int] = (2, 30),
        min_samples_leaf_bounds: tuple[int, int] = (1, 20),
        use_scaler: bool = False,
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

        # Bias toward parameters that usually change behavior more than
        # n_estimators (which often affects variance/runtime more than split shape).
        if r < 0.25:
            # n_estimators move
            step = int(rng.choice([-10, 10]))
            new_n_estimators = int(np.clip(new_n_estimators + step, ne_lo, ne_hi))

        elif r < 0.65:
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

        elif r < 0.85:
            # min_samples_leaf move
            step = int(rng.choice([-1, 1]))
            new_min_samples_leaf = int(np.clip(new_min_samples_leaf + step, msl_lo, msl_hi))

        else:
            # max_features toggle (simple, robust options)
            choices = ["sqrt", "log2", 0.5, 1.0]
            # pick a different one than current
            choices = [c for c in choices if c != new_max_features]
            # Choose by index to avoid NumPy coercing mixed types to strings (e.g. 0.5 -> "0.5")
            new_max_features = choices[int(rng.integers(0, len(choices)))]

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
      - neighbour(rng) -> dict with nearby hyperparams
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64,),   # keep 1-2 layers
        activation: str = "relu",                      # "relu" or "tanh"
        alpha: float = 1e-4,                           # L2 regularization
        learning_rate_init: float = 1e-2,
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

        if r < 0.70:
            # tweak architecture
            arch = list(new_arch)
            lo_l, hi_l = self.bounds_layers

            # with moderate prob, add/remove a layer (if allowed)
            if rng.random() < 0.35 and (lo_l != hi_l):
                if len(arch) < hi_l and rng.random() < 0.5:
                    insert_idx = int(rng.integers(0, len(arch) + 1)) if arch else 0
                    seed_units = arch[insert_idx] if (arch and insert_idx < len(arch)) else (arch[-1] if arch else 64)
                    arch.insert(insert_idx, seed_units)
                elif len(arch) > lo_l:
                    remove_idx = int(rng.integers(0, len(arch)))
                    arch.pop(remove_idx)

            # tweak one layer width +/- step
            if len(arch) == 0:
                arch = [64]
            idx = int(rng.integers(0, len(arch)))
            magnitude = int(rng.choice([8, 16, 32], p=[0.45, 0.40, 0.15]))
            step = magnitude if rng.random() < 0.5 else -magnitude
            arch[idx] = arch[idx] + step

            new_arch = self._clip_arch(tuple(arch))

        elif r < 0.82:
            # alpha log move
            a = self._clip_pos(new_alpha, self.alpha_bounds)
            new_alpha = self._clip_pos(np.exp(np.log(a) + rng.normal(0.0, self.log_step_alpha)), self.alpha_bounds)

        elif r < 0.90:
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