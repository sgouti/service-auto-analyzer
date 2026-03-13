#  Copyright 2023 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
from typing import Any, Optional

from typing_extensions import override

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - legacy compatibility path
    XGBClassifier = None

from app.commons import logging
from app.commons.object_saving.object_saver import ObjectSaver
from app.ml.models import MlModel
from app.utils import text_processing

LOGGER = logging.getLogger("analyzerApp.boosting_decision_maker")

MODEL_FILES: list[str] = ["boost_model.pickle", "data_features_config.pickle"]
DEFAULT_RANDOM_STATE = 43
DEFAULT_N_ESTIMATORS = 75
DEFAULT_MAX_DEPTH = 5
DEFAULT_LEARNING_RATE = 0.2
DEFAULT_MODEL_BACKEND = "lightgbm"
DEFAULT_NUM_LEAVES = 31
DEFAULT_MIN_CHILD_SAMPLES = 20
DEFAULT_SUBSAMPLE = 0.9
DEFAULT_COLSAMPLE_BYTREE = 0.9
DEFAULT_REG_LAMBDA = 0.0


class BoostingDecisionMaker(MlModel):
    _loaded: bool
    n_estimators: int
    max_depth: int
    random_state: int
    learning_rate: float
    feature_ids: list[int]
    monotonous_features: set[int]
    model_backend: str
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    boost_model: Any

    def __init__(
        self,
        object_saver: ObjectSaver,
        tags: str = "global boosting model",
        *,
        features: Optional[list[int]] = None,
        monotonous_features: Optional[list[int]] = None,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        learning_rate: Optional[float] = None,
        model_backend: str = DEFAULT_MODEL_BACKEND,
        num_leaves: Optional[int] = None,
        min_child_samples: Optional[int] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        reg_lambda: Optional[float] = None,
    ) -> None:
        super().__init__(object_saver, tags)
        self.n_estimators = n_estimators if n_estimators is not None else DEFAULT_N_ESTIMATORS
        self.max_depth = max_depth if max_depth is not None else DEFAULT_MAX_DEPTH
        self.random_state = random_state if random_state is not None else DEFAULT_RANDOM_STATE
        self.learning_rate = learning_rate if learning_rate is not None else DEFAULT_LEARNING_RATE
        self.model_backend = model_backend
        self.num_leaves = num_leaves if num_leaves is not None else DEFAULT_NUM_LEAVES
        self.min_child_samples = min_child_samples if min_child_samples is not None else DEFAULT_MIN_CHILD_SAMPLES
        self.subsample = subsample if subsample is not None else DEFAULT_SUBSAMPLE
        self.colsample_bytree = colsample_bytree if colsample_bytree is not None else DEFAULT_COLSAMPLE_BYTREE
        self.reg_lambda = reg_lambda if reg_lambda is not None else DEFAULT_REG_LAMBDA
        self.boost_model = self._build_classifier()
        self.feature_ids = features if features else []
        self.monotonous_features = set(monotonous_features) if monotonous_features else set()
        self._loaded = False

    def _build_classifier(self, **overrides: Any) -> Any:
        backend = overrides.pop("model_backend", self.model_backend)
        monotone_constraints = overrides.pop("monotone_constraints", None)
        classifier_kwargs = {
            "n_estimators": overrides.pop("n_estimators", self.n_estimators),
            "max_depth": overrides.pop("max_depth", self.max_depth),
            "random_state": overrides.pop("random_state", self.random_state),
            "learning_rate": overrides.pop("learning_rate", self.learning_rate),
            "num_leaves": overrides.pop("num_leaves", self.num_leaves),
            "min_child_samples": overrides.pop("min_child_samples", self.min_child_samples),
            "subsample": overrides.pop("subsample", self.subsample),
            "colsample_bytree": overrides.pop("colsample_bytree", self.colsample_bytree),
            "reg_lambda": overrides.pop("reg_lambda", self.reg_lambda),
        }
        if backend == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is required to load legacy boosting models")
            xgb_kwargs = {
                "n_estimators": classifier_kwargs["n_estimators"],
                "max_depth": classifier_kwargs["max_depth"],
                "random_state": classifier_kwargs["random_state"],
                "learning_rate": classifier_kwargs["learning_rate"],
            }
            if monotone_constraints is not None:
                xgb_kwargs["monotone_constraints"] = monotone_constraints
            return XGBClassifier(**xgb_kwargs)

        lgbm_kwargs = {
            **classifier_kwargs,
            "objective": "binary",
            "verbosity": -1,
            "class_weight": "balanced",
        }
        if monotone_constraints is not None:
            lgbm_kwargs["monotone_constraints"] = monotone_constraints
        return LGBMClassifier(**lgbm_kwargs)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    @override
    def is_custom(self) -> bool:
        """Indicates whether the model is custom or not."""
        return False

    @property
    def feature_importances(self) -> Optional[dict[int, float]]:
        if self.loaded:
            importances = np.asarray(getattr(self.boost_model, "feature_importances_", []), dtype=float).tolist()
            return dict(zip(self.feature_ids, importances))
        return None

    def load_model(self) -> None:
        if self.loaded:
            return
        boost_model, features_config = self._load_models(MODEL_FILES)
        if len(boost_model) > 4 and isinstance(boost_model[0], str):
            (
                self.model_backend,
                self.n_estimators,
                self.max_depth,
                self.random_state,
                self.learning_rate,
                self.num_leaves,
                self.min_child_samples,
                self.subsample,
                self.colsample_bytree,
                self.reg_lambda,
                self.boost_model,
            ) = boost_model
            self.feature_ids, self.monotonous_features = features_config
        elif len(boost_model) > 3:
            # New model format
            self.n_estimators, self.max_depth, self.random_state, self.boost_model = boost_model
            self.learning_rate = getattr(self.boost_model, "learning_rate", self.learning_rate)
            self.model_backend = (
                "xgboost"
                if XGBClassifier is not None and isinstance(self.boost_model, XGBClassifier)
                else DEFAULT_MODEL_BACKEND
            )
            self.num_leaves = getattr(self.boost_model, "num_leaves", self.num_leaves)
            self.min_child_samples = getattr(self.boost_model, "min_child_samples", self.min_child_samples)
            self.subsample = getattr(self.boost_model, "subsample", self.subsample)
            self.colsample_bytree = getattr(self.boost_model, "colsample_bytree", self.colsample_bytree)
            self.reg_lambda = getattr(self.boost_model, "reg_lambda", self.reg_lambda)
            self.feature_ids, self.monotonous_features = features_config
        else:
            # Old model format
            self.n_estimators, self.max_depth, self.boost_model = boost_model
            self.random_state = DEFAULT_RANDOM_STATE
            self.learning_rate = getattr(self.boost_model, "learning_rate", self.learning_rate)
            self.model_backend = (
                "xgboost"
                if XGBClassifier is not None and isinstance(self.boost_model, XGBClassifier)
                else DEFAULT_MODEL_BACKEND
            )
            _, features, self.monotonous_features = features_config
            self.feature_ids = text_processing.transform_string_feature_range_into_list(features)
        self._loaded = True

    def save_model(self):
        self._save_models(
            zip(
                MODEL_FILES,
                [
                    [
                        self.model_backend,
                        self.n_estimators,
                        self.max_depth,
                        self.random_state,
                        self.learning_rate,
                        self.num_leaves,
                        self.min_child_samples,
                        self.subsample,
                        self.colsample_bytree,
                        self.reg_lambda,
                        self.boost_model,
                    ],
                    [self.feature_ids, self.monotonous_features],
                ],
            )
        )

    def train_model(self, train_data: list[list[float]], labels: list[int]) -> float:
        mon_features = [1 if feature in self.monotonous_features else 0 for feature in self.feature_ids]
        monotone_constraints: str | list[int] | None
        monotone_constraints = mon_features if any(mon_features) else None
        if self.model_backend == "xgboost" and monotone_constraints is not None:
            monotone_constraints = "(" + ",".join([str(f) for f in mon_features]) + ")"
        self.boost_model = self._build_classifier(monotone_constraints=monotone_constraints)
        self.boost_model.fit(train_data, labels)
        self._loaded = True
        res = self.boost_model.predict(train_data)
        f1 = f1_score(y_pred=res, y_true=labels)
        if f1 is None:
            f1 = 0.0
        LOGGER.debug(f"Train dataset F1 score: {f1:.5f}")
        LOGGER.debug(
            "Feature importances: %s",
            json.dumps(self.feature_importances),
        )
        return f1

    def tune_hyperparameters(self, train_data: list[list[float]], labels: list[int], trials: int) -> float:
        if not train_data or not labels:
            return 0.0
        try:
            import optuna
        except ImportError:
            LOGGER.warning("Optuna is not installed; skipping hyperparameter tuning")
            return 0.0

        x_train, x_valid, y_train, y_valid = train_test_split(
            train_data,
            labels,
            test_size=0.2,
            random_state=self.random_state,
            stratify=labels,
        )
        mon_features = [1 if feature in self.monotonous_features else 0 for feature in self.feature_ids]

        def objective(trial: Any) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
            classifier = self._build_classifier(
                monotone_constraints=mon_features if any(mon_features) else None,
                **params,
            )
            classifier.fit(x_train, y_train)
            predicted = classifier.predict(x_valid)
            score = f1_score(y_valid, predicted)
            return 0.0 if score is None else float(score)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=max(1, trials), show_progress_bar=False)
        best_params = study.best_params
        self.n_estimators = int(best_params.get("n_estimators", self.n_estimators))
        self.max_depth = int(best_params.get("max_depth", self.max_depth))
        self.learning_rate = float(best_params.get("learning_rate", self.learning_rate))
        self.num_leaves = int(best_params.get("num_leaves", self.num_leaves))
        self.min_child_samples = int(best_params.get("min_child_samples", self.min_child_samples))
        self.subsample = float(best_params.get("subsample", self.subsample))
        self.colsample_bytree = float(best_params.get("colsample_bytree", self.colsample_bytree))
        self.reg_lambda = float(best_params.get("reg_lambda", self.reg_lambda))
        LOGGER.info("Applied Optuna-tuned LightGBM params: %s", json.dumps(best_params))
        return float(study.best_value)

    def predict(self, data: list[list[float]]) -> tuple[list[int], list[list[float]]]:
        if not len(data):
            return [], []
        return self.boost_model.predict(data).tolist(), self.boost_model.predict_proba(data).tolist()

    def validate_model(self, valid_test_set: list[list[float]], valid_test_labels: list[int]) -> float:
        res, _ = self.predict(valid_test_set)
        f1 = f1_score(y_pred=res, y_true=valid_test_labels)
        if f1 is None:
            f1 = 0.0
        LOGGER.debug(f"Valid dataset F1 score: {f1:.5f}")
        LOGGER.debug(f"\n{confusion_matrix(valid_test_labels, res)}")
        LOGGER.debug(f"\n{classification_report(valid_test_labels, res)}")
        return f1
