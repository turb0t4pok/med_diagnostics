import os
from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # НОВОЕ
from sklearn.linear_model import LogisticRegression  # НОВОЕ
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score,
    f1_score,
)

import joblib


class MedicalRecommenderApp:
    """
    Класс, инкапсулирующий логику работы прототипа
    рекомендательной системы для медицинской диагностики.
    """

    def __init__(self) -> None:
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.target_name: Optional[str] = None

        # Храним данные для сравнения моделей (НОВОЕ)
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    @staticmethod
    def _input_non_empty(prompt: str) -> str:
        """Запрос строки, пока не будет введено непустое значение."""
        while True:
            value = input(prompt).strip()
            if value:
                return value
            print("Поле не может быть пустым. Повторите ввод.")

    def load_data_and_train(self) -> None:
        """
        Загрузка датасета из CSV, подготовка данных,
        разбиение на обучающую и тестовую выборки и обучение модели (GradientBoosting).
        """
        print("\n=== ЗАГРУЗКА ДАННЫХ И ОБУЧЕНИЕ МОДЕЛИ ===")

        path = self._input_non_empty(
            "Введите путь к CSV-файлу с данными (например, medical_sample_ru.csv): "
        )

        if not os.path.exists(path):
            print(f"Файл '{path}' не найден.")
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            return

        if df.empty:
            print("Файл пустой или не содержит данных.")
            return

        print("\nПервые строки датасета:")
        print(df.head())
        print("\nСписок столбцов:")
        for col in df.columns:
            print(f" - {col}")

        target = self._input_non_empty(
            "\nВведите имя столбца, содержащего диагноз (целевой класс) "
            "(например, 'диагноз'): "
        )

        if target not in df.columns:
            print(f"Столбец '{target}' не найден в данных.")
            return

        self.target_name = target

        X = df.drop(columns=[target])
        y = df[target]

        if X.empty:
            print("После удаления целевого столбца не осталось признаков.")
            return

        self.numeric_features = X.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()
        self.categorical_features = [c for c in X.columns if c not in self.numeric_features]

        self.feature_names = list(X.columns)

        print("\nОпределённые типы признаков:")
        print("Числовые признаки:", self.numeric_features or "нет")
        print("Категориальные признаки:", self.categorical_features or "нет")

        # Препроцессинг для числовых и категориальных признаков
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if len(y.unique()) > 1 else None,
            )
        except ValueError as e:
            print(f"Ошибка при разбиении на обучающую и тестовую выборки: {e}")
            return

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        classifier = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        print("\nИдёт обучение модели (GradientBoosting с балансировкой классов)...")
        self.model.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        print("Обучение завершено.")

        y_pred_train = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        print(f"Точность на обучающей выборке: {train_acc:.3f}")

        y_pred_test = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"Точность на тестовой выборке: {test_acc:.3f}")

        try:
            gb = self.model.named_steps["classifier"]
            preprocessor = self.model.named_steps["preprocessor"]
            transformed_feature_names = preprocessor.get_feature_names_out()
            importances = gb.feature_importances_

            original_importance = {name: 0.0 for name in self.feature_names}
            for t_name, imp in zip(transformed_feature_names, importances):
                if '__' in t_name:
                    base = t_name.split('__')[1]
                    if base in original_importance:
                        original_importance[base] += imp
                    else:
                        base_clean = base.split('_')[0]
                        if base_clean in original_importance:
                            original_importance[base_clean] += imp
                        else:
                            for orig in original_importance:
                                if base.startswith(orig):
                                    original_importance[orig] += imp
                                    break
            sorted_imp = sorted(original_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\n=== ГЛОБАЛЬНАЯ ВАЖНОСТЬ ПРИЗНАКОВ (топ-10) ===")
            for name, imp in sorted_imp:
                print(f"  {name}: {imp:.4f}")
        except Exception as e:
            print(f"Не удалось вычислить глобальную важность признаков: {e}")

    def evaluate_model(self) -> None:
        """
        Оценка качества обученной модели на тестовой выборке
        (accuracy, классификационный отчёт, top-k accuracy)
        + вывод нескольких примеров из тестовой выборки.
        """
        print("\n=== ОЦЕНКА КАЧЕСТВА МОДЕЛИ ===")

        if self.model is None or self.X_test is None or self.y_test is None:
            print(
                "Модель и/или тестовая выборка отсутствуют.\n"
                "Сначала загрузите данные и обучите модель (пункт 1 меню)."
            )
            return

        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"\nТочность (accuracy) на тестовой выборке: {acc:.3f}\n")

        print("Классификационный отчёт по классам (precision, recall, F1):")
        print(classification_report(self.y_test, y_pred, zero_division=0, digits=3))

        y_proba = None
        classes = None
        if hasattr(self.model, "predict_proba"):
            try:
                y_proba = self.model.predict_proba(self.X_test)
                classes = self.model.named_steps["classifier"].classes_
                k = min(3, len(classes))
                topk = top_k_accuracy_score(self.y_test, y_proba, k=k, labels=classes)
                print(f"Доля правильного диагноза в топ-{k} рекомендаций: {topk:.3f}")
            except Exception as e:
                print(f"Не удалось вычислить top-k accuracy: {e}")
        else:
            print(
                "Модель не поддерживает вероятности (predict_proba), "
                "поэтому top-k accuracy не вычисляется."
            )

        print("\n=== ПРИМЕРЫ СЛУЧАЕВ ИЗ ТЕСТОВОЙ ВЫБОРКИ ===")

        n_examples_default = 5
        try:
            raw_n = input(
                f"Сколько случаев показать? (по умолчанию {n_examples_default}): "
            ).strip()
            n_examples = int(raw_n) if raw_n else n_examples_default
        except ValueError:
            print("Некорректное число, использую значение по умолчанию.")
            n_examples = n_examples_default

        if n_examples <= 0:
            n_examples = n_examples_default

        n_examples = min(n_examples, len(self.X_test))

        if n_examples == 0:
            print("Тестовая выборка пуста, примеры показать нельзя.")
            return

        X_sample = self.X_test.iloc[:n_examples]
        y_true_sample = self.y_test.iloc[:n_examples]
        y_pred_sample = y_pred[:n_examples]

        if y_proba is not None and classes is not None:
            y_proba_sample = y_proba[:n_examples]
        else:
            y_proba_sample = None

        records = X_sample.to_dict(orient="records")

        for i, (idx, features, true_label, pred_label) in enumerate(
            zip(X_sample.index, records, y_true_sample, y_pred_sample),
            start=1
        ):
            print(f"\nСлучай {i} (индекс в датасете: {idx}):")
            print("  Признаки:")
            for feat_name, feat_value in features.items():
                print(f"    {feat_name}: {feat_value}")

            print(f"  Истинный диагноз:      {true_label}")
            print(f"  Предсказанный диагноз: {pred_label}")

            if y_proba_sample is not None:
                prob_row = y_proba_sample[i - 1]
                k = min(3, len(classes))
                top_idx = np.argsort(prob_row)[::-1][:k]
                print("  Топ рекомендаций:")
                for rank, cls_idx in enumerate(top_idx, start=1):
                    cls_name = classes[cls_idx]
                    p = prob_row[cls_idx]
                    print(f"    {rank}. {cls_name} (вероятность = {p:.3f})")

    def compare_models(self) -> None:
        """
        Сравнение нескольких моделей классификации на одних и тех же данных.
        Доступно только после загрузки и обучения модели (пункт 1).
        """
        print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
        if self.X_train is None or self.X_test is None:
            print("Данные отсутствуют. Сначала выполните пункт 1 (загрузка и обучение модели).")
            return

        print("Выбранные модели:")
        print(" - Случайный лес (RandomForest) с class_weight='balanced'")
        print(" - Логистическая регрессия (LogisticRegression) с max_iter=1000")
        print(" - Градиентный бустинг (GradientBoostingClassifier)")

        # Определяем препроцессор (такой же, как в основной модели)
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)
            ]
        )

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced'
            ),
            "LogisticRegression": LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced', n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200, random_state=42
            )
        }

        results = []
        classes = None

        for name, clf in models.items():
            print(f"\nОбучение {name}...")
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            # Macro F1 (среднее по классам)
            f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
            # Top-3 accuracy
            if hasattr(pipeline, "predict_proba"):
                try:
                    y_proba = pipeline.predict_proba(self.X_test)
                    classes = pipeline.named_steps["classifier"].classes_
                    k = min(3, len(classes))
                    top3 = top_k_accuracy_score(self.y_test, y_proba, k=k, labels=classes)
                except:
                    top3 = 0.0
            else:
                top3 = 0.0
            results.append({
                "Модель": name,
                "Accuracy": acc,
                "Macro F1": f1_macro,
                f"Top-{min(3, len(classes) if classes is not None else 3)} accuracy": top3
            })

        print("\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        print("\nПримечание: Macro F1 — среднее F1-меры по всем классам (учитывает несбалансированность).")

    def _ask_features_from_user(self) -> Optional[pd.DataFrame]:
        """
        Запрос значений признаков у пользователя.
        Возвращает DataFrame с одним клиническим случаем,
        либо None при ошибке.
        """
        if not self.feature_names:
            print("Список признаков неизвестен. Сначала обучите модель.")
            return None

        print("\nВвод данных нового клинического случая.")
        print("Подсказка: для любого признака можно просто нажать Enter, "
              "чтобы оставить его пустым (пропуск).")
        print("Для бинарных признаков используйте 0/1 (0 — нет, 1 — да).")
        print("Например, 'пол' — М/Ж, 'симптом_...' — 0 или 1.\n")

        values = {}
        for feature in self.feature_names:
            if feature in self.numeric_features:
                while True:
                    raw = input(f"{feature} (число, пусто — пропуск): ").strip()
                    if raw == "":
                        values[feature] = np.nan
                        break
                    try:
                        values[feature] = float(raw.replace(",", "."))
                        break
                    except ValueError:
                        print("Некорректное числовое значение, введите число ещё раз.")
            else:
                raw = input(f"{feature} (категория/текст, пусто — пропуск): ").strip()
                if raw == "":
                    values[feature] = np.nan
                else:
                    values[feature] = raw

        X_new = pd.DataFrame([values])
        return X_new

    def recommend_for_new_case(self) -> None:
        """
        Получение диагностических рекомендаций для нового клинического случая.
        """
        print("\n=== ПОЛУЧЕНИЕ ДИАГНОСТИЧЕСКИХ РЕКОМЕНДАЦИЙ ===")

        if self.model is None:
            print("Модель не обучена. Сначала загрузите данные и обучите модель (пункт 1).")
            return

        X_new = self._ask_features_from_user()
        if X_new is None:
            return

        try:
            y_pred = self.model.predict(X_new)[0]
            print(f"\nНаиболее вероятный диагноз (по модели): {y_pred}")
        except Exception as e:
            print(f"Ошибка при получении диагноза: {e}")
            return

        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X_new)[0]
                classes = self.model.named_steps["classifier"].classes_
                idx_sorted = np.argsort(proba)[::-1]
                k = min(3, len(classes))
                print("\nТоп диагностических рекомендаций:")
                for rank in range(k):
                    cls = classes[idx_sorted[rank]]
                    p = proba[idx_sorted[rank]]
                    print(f"{rank + 1}. {cls} (вероятность = {p:.3f})")
            except Exception as e:
                print(f"Не удалось вычислить вероятности для топ-рекомендаций: {e}")

    def save_model(self) -> None:
        """
        Сохранение обученной модели и служебной информации в файл.
        """
        print("\n=== СОХРАНЕНИЕ МОДЕЛИ ===")

        if self.model is None:
            print("Модель не обучена. Сначала загрузите данные и обучите модель.")
            return

        filename = input(
            "Введите имя файла для сохранения (по умолчанию model_med.joblib): "
        ).strip()
        if not filename:
            filename = "model_med.joblib"

        state = {
            "model": self.model,
            "feature_names": self.feature_names,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "target_name": self.target_name,
            # Сохраняем также данные для сравнения (НОВОЕ)
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
        }

        try:
            joblib.dump(state, filename)
            print(f"Модель и параметры сохранены в файл '{filename}'.")
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")

    def load_model(self) -> None:
        """
        Загрузка модели и служебной информации из файла.
        """
        print("\n=== ЗАГРУЗКА МОДЕЛИ ===")
        filename = self._input_non_empty(
            "Введите имя файла с моделью (например, model_med.joblib): "
        )

        if not os.path.exists(filename):
            print(f"Файл '{filename}' не найден.")
            return

        try:
            state = joblib.load(filename)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            return

        try:
            self.model = state["model"]
            self.feature_names = state["feature_names"]
            self.numeric_features = state["numeric_features"]
            self.categorical_features = state["categorical_features"]
            self.target_name = state["target_name"]
            # Загружаем данные, если они есть в файле (НОВОЕ)
            self.X_train = state.get("X_train", None)
            self.X_test = state.get("X_test", None)
            self.y_train = state.get("y_train", None)
            self.y_test = state.get("y_test", None)
        except KeyError:
            print("Структура файла модели некорректна.")
            return

        print("Модель успешно загружена. "
              "Для расчёта метрик потребуется заново загрузить данные и выполнить оценку.")

    def run(self) -> None:
        """
        Основной цикл приложения с консольным меню.
        """
        while True:
            print("\n============================================")
            print(" ПРОТОТИП РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ ДИАГНОСТИКИ ")
            print("============================================")
            print("1. Загрузить данные и обучить модель")
            print("2. Оценить качество модели на тестовой выборке")
            print("3. Получить рекомендации для нового клинического случая")
            print("4. Сохранить обученную модель в файл")
            print("5. Загрузить модель из файла")
            print("6. Сравнить модели (RandomForest, LogisticRegression, GradientBoosting)")  # НОВОЕ
            print("0. Выход")
            choice = input("Выберите пункт меню и нажмите Enter: ").strip()

            if choice == "1":
                self.load_data_and_train()
            elif choice == "2":
                self.evaluate_model()
            elif choice == "3":
                self.recommend_for_new_case()
            elif choice == "4":
                self.save_model()
            elif choice == "5":
                self.load_model()
            elif choice == "6":   # НОВОЕ
                self.compare_models()
            elif choice == "0":
                print("Завершение работы приложения.")
                break
            else:
                print("Неизвестный пункт меню, повторите ввод.")


def main() -> None:
    app = MedicalRecommenderApp()
    app.run()


if __name__ == "__main__":
    main()