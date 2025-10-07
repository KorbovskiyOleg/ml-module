import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

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

# --- Загружаем данные ---
def load_zoo_data():
    query = "SELECT * FROM zoo_training_data;"
    df = pd.read_sql(query, engine)
    print(f"📦 Загружено {len(df)} строк из таблицы zoo_training_data")

    # Убедимся, что есть нужные столбцы
    print(f"📊 Колонки: {list(df.columns)}")

    # Удаляем нечисловую колонку animal_name
    if "animal_name" in df.columns:
        df = df.drop(columns=["animal_name"])

    # Предположим, что у тебя последний столбец — это класс животного (label)
    label_col = "animal_type"  # если у тебя он называется по-другому — поправь!
    x = df.drop(columns=[label_col])
    y = df[label_col]

    return x, y

# --- Обучаем модель ---
def train_zoo_model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Точность модели: {acc:.3f}")
    print("🔍 Отчёт о классификации:\n", classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/zoo_model.pkl")
    print("💾 Модель сохранена в models/zoo_model.pkl")

# --- Основной запуск ---
def main():
    x, y = load_zoo_data()
    train_zoo_model(x, y)

if __name__ == "__main__":
    main()
