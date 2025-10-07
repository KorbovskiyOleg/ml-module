from sqlalchemy import create_engine, text, Table, Column, Integer, String, MetaData, Float
from dotenv import load_dotenv
import os

# --- Загружаем переменные окружения ---
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "ml_data")

# --- Создаём подключение к PostgresSQL ---
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={"options": f"-csearch_path={DB_SCHEMA}"}
)

# --- Проверка соединения ---
def test_connection():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT current_schema();"))
        print("✅ Connected to schema:", res.scalar())

# --- Создание схемы, если её нет ---
def ensure_schema_exists():
    with engine.begin() as conn:  # важно: begin → автокоммит
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};"))
    print(f"📂 Schema '{DB_SCHEMA}' is ready.")

# --- Создание таблицы training_data ---
def create_training_table():
    metadata = MetaData(schema=DB_SCHEMA)

    training_table = Table(
        "training_data",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("feature_1", Float),
        Column("feature_2", Float),
        Column("feature_3", Float),
        Column("label", String(50))
    )

    metadata.create_all(engine)
    print("✅ Table 'training_data' is ready in schema:", DB_SCHEMA)

    #--- Вставка тестовых данных--
def insert_sample_data():
        sample_rows = [
            {"feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4, "label": 'cat'},
            {"feature_1": 7.0, "feature_2": 3.2, "feature_3": 4.7, "label": 'dog'},
            {"feature_1": 6.3, "feature_2": 3.3, "feature_3": 6.0, "label": 'horse'},

        ]

        with engine.begin() as conn:
            # очищаем таблицу
            conn.execute(text("TRUNCATE TABLE training_data RESTART IDENTITY;"))
            for row in sample_rows:
                conn.execute(
                    text("""
                        INSERT INTO training_data (feature_1, feature_2, feature_3, label)
                        VALUES (:f1, :f2, :f3, :label)
                    """),
                    {"f1": row["feature_1"], "f2": row["feature_2"], "f3": row["feature_3"], "label": row["label"]}
                )
        print(f"🐍 Inserted {len(sample_rows)} sample rows into training_data.")

def show_sample_data():
        with engine.connect() as conn:
            res = conn.execute(text("SELECT * FROM training_data LIMIT 5;"))
            rows = res.fetchall()
            if rows:
                print(f"✅ Данные успешно загружены в таблицу 'training_data'. ({len(rows)} строк)")
            else:
                print("⚠️ Таблица создана, но данные отсутствуют.")


# --- Проверка всех таблиц в схеме ---
def list_tables():
    with engine.connect() as conn:
        res = conn.execute(
            text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema
                ORDER BY table_name;
            """),
            {"schema": DB_SCHEMA}
        )
        tables = [r[0] for r in res]
        print("📋 Tables in schema", DB_SCHEMA, ":", tables if tables else "No tables found")

# --- Главный запуск ---
if __name__ == "__main__":
    ensure_schema_exists()
    test_connection()
    create_training_table()
    insert_sample_data()
    show_sample_data()
    list_tables()
