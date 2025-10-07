import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --- Загружаем .env ---
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "ml_data")

# --- Подключаемся к базе ---
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={"options": f"-csearch_path={DB_SCHEMA}"}
)


def analyze_dataset(x, y):
    """Расширенный анализ датасета"""
    print("\n" + "=" * 60)
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ ДАТАСЕТА")
    print("=" * 60)

    print(f"🔢 Общий размер данных: {len(x)} строк")
    print(f"🎯 Количество признаков: {x.shape[1]}")
    print(f"🏷️  Количество классов: {y.nunique()}")
    print(f"📈 Распределение классов:")
    class_distribution = y.value_counts().sort_index()
    for class_label, count in class_distribution.items():
        percentage = (count / len(y)) * 100
        print(f"   Класс {class_label}: {count} samples ({percentage:.1f}%)")

    # Проверка на пропущенные значения
    missing_values = x.isnull().sum()
    if missing_values.any():
        print(f"\n⚠️  Пропущенные значения:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"   {col}: {missing} пропусков")
    else:
        print(f"\n✅ Пропущенных значений нет")

    # Проверка на дубликаты
    duplicates = x.duplicated().sum()
    print(f"\n🔍 Дубликаты в данных: {duplicates} ({duplicates / len(x) * 100:.1f}%)")

    if duplicates > len(x) * 0.1:  # Если больше 10% дубликатов
        print("⚠️  ВНИМАНИЕ: Много дубликатов! Это может влиять на качество модели.")


def compare_models(x, y):
    """Сравнение Decision Tree и Random Forest"""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 Разделение данных:")
    print(f"   Обучающая выборка: {x_train.shape[0]} samples")
    print(f"   Тестовая выборка: {x_test.shape[0]} samples")

    # Проверяем распределение классов в тестовой выборке
    print(f"\n📋 Распределение классов в тестовой выборке:")
    test_distribution = y_test.value_counts().sort_index()
    for class_label, count in test_distribution.items():
        print(f"   {class_label}: {count} samples")

    # Decision Tree
    dt_model = DecisionTreeClassifier(
        criterion='entropy',
        random_state=42,
        max_depth=10
    )

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=10
    )

    print("\n" + "🔄 ОБУЧЕНИЕ МОДЕЛЕЙ" + "🔁" * 20)

    # Обучаем модели
    dt_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)

    # Предсказания
    dt_pred = dt_model.predict(x_test)
    rf_pred = rf_model.predict(x_test)

    # Метрики
    dt_acc = accuracy_score(y_test, dt_pred)
    rf_acc = accuracy_score(y_test, rf_pred)

    print("\n" + "=" * 60)
    print("📈 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    print(f"🌳 Decision Tree Accuracy:  {dt_acc:.4f} ({dt_acc * 100:.2f}%)")
    print(f"🌲 Random Forest Accuracy: {rf_acc:.4f} ({rf_acc * 100:.2f}%)")

    # Анализ различий в предсказаниях
    different_predictions = sum(dt_pred != rf_pred)
    print(f"🔀 Различающихся предсказаний: {different_predictions}/{len(y_test)}")

    # Важность признаков (текстовый вывод)
    print(f"\n🔝 Топ-5 важных признаков (Random Forest):")
    feature_importances = pd.DataFrame({
        'feature': x.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in feature_importances.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

    return dt_model, rf_model, x_train, x_test, y_train, y_test, dt_pred, rf_pred


def analyze_class_performance(y_test, dt_pred, rf_pred, class_names):
    """Анализ производительности по классам"""
    print("\n" + "=" * 60)
    print("🎯 АНАЛИЗ ПО КЛАССАМ")
    print("=" * 60)

    for class_name in class_names:
        class_mask = y_test == class_name
        if sum(class_mask) > 0:
            dt_correct = sum((y_test[class_mask] == dt_pred[class_mask]))
            rf_correct = sum((y_test[class_mask] == rf_pred[class_mask]))
            total = sum(class_mask)

            print(f"\n{class_name}: {total} samples")
            print(f"   Decision Tree: {dt_correct}/{total} correct ({dt_correct / total * 100:.1f}%)")
            print(f"   Random Forest: {rf_correct}/{total} correct ({rf_correct / total * 100:.1f}%)")


def print_confusion_analysis(y_test, predictions, model_name):
    """Текстовый анализ матрицы ошибок"""
    cm = confusion_matrix(y_test, predictions)
    classes = sorted(y_test.unique())

    print(f"\n🔍 {model_name} - Анализ ошибок:")
    for i, true_class in enumerate(classes):
        total = cm[i].sum()
        correct = cm[i][i]
        errors = total - correct

        if errors > 0:
            error_details = []
            for j, pred_class in enumerate(classes):
                if i != j and cm[i][j] > 0:
                    error_details.append(f"{pred_class}({cm[i][j]})")

            if error_details:
                print(f"   {true_class}: {errors} ошибок → {', '.join(error_details)}")


# --- Загружаем данные ---
def load_zoo_data():
    query = "SELECT * FROM zoo_training_data;"
    df = pd.read_sql(query, engine)
    print(f"📦 Загружено {len(df)} строк из таблицы zoo_training_data")
    print(f"📊 Колонки: {list(df.columns)}")

    # Удаляем нечисловую колонку animal_name
    if "animal_name" in df.columns:
        df = df.drop(columns=["animal_name"])

    label_col = "animal_type"
    x = df.drop(columns=[label_col])
    y = df[label_col]

    return x, y


# --- Основной запуск ---
def main():
    print("🚀 ЗАПУСК АНАЛИЗА МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
    print("=" * 60)

    # Загрузка данных
    x, y = load_zoo_data()

    # Анализ датасета
    analyze_dataset(x, y)

    # Сравнение моделей
    dt_model, rf_model, x_train, x_test, y_train, y_test, dt_pred, rf_pred = compare_models(x, y)

    # Детальные отчеты
    print("\n" + "=" * 60)
    print("🔍 ДЕТАЛЬНЫЕ ОТЧЕТЫ КЛАССИФИКАЦИИ")
    print("=" * 60)

    print("\nDecision Tree:")
    print(classification_report(y_test, dt_pred, zero_division=0))

    print("\nRandom Forest:")
    print(classification_report(y_test, rf_pred, zero_division=0))

    # Анализ по классам
    analyze_class_performance(y_test, dt_pred, rf_pred, sorted(y.unique()))

    # Анализ ошибок
    print_confusion_analysis(y_test, dt_pred, "Decision Tree")
    print_confusion_analysis(y_test, rf_pred, "Random Forest")

    # Сохранение моделей
    os.makedirs("models", exist_ok=True)
    joblib.dump(dt_model, "models/zoo_decision_tree.pkl")
    joblib.dump(rf_model, "models/zoo_random_forest.pkl")

    print(f"\n💾 Модели сохранены:")
    print(f"   • Decision Tree: models/zoo_decision_tree.pkl")
    print(f"   • Random Forest: models/zoo_random_forest.pkl")

    # Рекомендации
    print("\n" + "=" * 60)
    print("💡 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ")
    print("=" * 60)
    print("1. 🎯 Решение проблемы редких классов:")
    print("   • Добавьте больше данных для reptile, amphibian")
    print("   • Используйте oversampling (SMOTE)")
    print("   • Примените взвешивание классов (class_weight='balanced')")

    print("\n2. 🔧 Улучшение качества данных:")
    print("   • Исследуйте и удалите дубликаты (42% данных!)")
    print("   • Проверьте корректность разметки")

    print("\n3. 🚀 Дальнейшие эксперименты:")
    print("   • Попробуйте Gradient Boosting (XGBoost, LightGBM)")
    print("   • Используйте кросс-валидацию")
    print("   • Поэкспериментируйте с гиперпараметрами")


if __name__ == "__main__":
    main()
