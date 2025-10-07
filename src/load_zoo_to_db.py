# src/load_zoo_to_db.py
from scipy.io import arff
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Boolean
from sqlalchemy.sql import text
from dotenv import load_dotenv
import os

# ---------- Настройки из .env ----------
load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "cardb")
DB_SCHEMA = os.getenv("DB_SCHEMA", "ml_data")

ARFF_PATH = os.path.join("data", "zoo.arff")  # путь к файлу .arff

# ---------- Создаём engine (search_path = DB_SCHEMA) ----------
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={"options": f"-csearch_path={DB_SCHEMA}"}
)

# ---------- Вспомогательные функции ----------
def ensure_schema_exists():
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};"))
    print(f"📂 Schema '{DB_SCHEMA}' is ready.")

def create_zoo_table():
    metadata = MetaData(schema=DB_SCHEMA)

    zoo_table = Table(
        "zoo_training_data",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("animal_name", String(100), nullable=False),
        Column("hair", Boolean),
        Column("feathers", Boolean),
        Column("eggs", Boolean),
        Column("milk", Boolean),
        Column("airborne", Boolean),
        Column("aquatic", Boolean),
        Column("predator", Boolean),
        Column("toothed", Boolean),
        Column("backbone", Boolean),
        Column("breathes", Boolean),
        Column("venomous", Boolean),
        Column("fins", Boolean),
        Column("legs", Integer),
        Column("tail", Boolean),
        Column("domestic", Boolean),
        Column("catsize", Boolean),
        Column("animal_type", String(50), nullable=False)
    )

    metadata.create_all(engine)
    print("✅ Table 'zoo_training_data' is ready in schema:", DB_SCHEMA)

def read_arff_to_df(arff_path):
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)

    # деккодиррование байтов в строки для столбцов объекта
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, (bytes, bytearray)) else x)

    # булевые столбцы в dataset
    bool_cols = [
        "hair","feathers","eggs","milk","airborne","aquatic","predator",
        "toothed","backbone","breathes","venomous","fins","tail","domestic","catsize"
    ]
    for c in bool_cols:
        if c in df.columns:
            # values are strings 'true'/'false' or already bool-like; normalize:
            df[c] = df[c].apply(lambda v: True if str(v).lower() == "true" else (False if str(v).lower() == "false" else None))

    # проверка что число ног является integer
    if "legs" in df.columns:
        df["legs"] = df["legs"].astype(int)

    # rename animal/type columns to match table
    if "animal" in df.columns:
        df = df.rename(columns={"animal": "animal_name"})
    if "type" in df.columns:
        df = df.rename(columns={"type": "animal_type"})

    # select only columns that we will store (avoid extra attrs)
    desired_cols = [
        "animal_name", "hair","feathers","eggs","milk","airborne","aquatic","predator",
        "toothed","backbone","breathes","venomous","fins","legs","tail","domestic","catsize",
        "animal_type"
    ]
    # keep only those available:
    cols_to_use = [c for c in desired_cols if c in df.columns]
    df = df[cols_to_use]

    return df

def write_df_to_db(df):
    # write to SQL: replace table contents
    # note: pandas.to_sql will create table if doesn't exist; we already created schema/table shape,
    # but using if_exists='replace' will recreate table matching pandas types.
    # To keep SQLAlchemy table definition exactly, you can use if_exists='append' after truncating.
    # Here we'll use replace for simplicity.
    df.to_sql("zoo_training_data", engine, schema=DB_SCHEMA, if_exists="replace", index=False)
    print(f"✅ DataFrame written to {DB_SCHEMA}.zoo_training_data (rows: {len(df)})")

def verify_count():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT COUNT(*) FROM zoo_training_data;"))
        count = res.scalar()
        print("🔢 Rows in zoo_training_data:", count)
        return count

# ---------- Основной запуск ----------
if __name__ == "__main__":
    if not os.path.exists(ARFF_PATH):
        raise SystemExit(f"ARFF file not found at {ARFF_PATH}")

    ensure_schema_exists()
    create_zoo_table()                 # создаст таблицу (пустую) если надо
    df = read_arff_to_df(ARFF_PATH)    # читаем и подготавливаем DataFrame
    # Записываем: заменим старую таблицу на новую из DataFrame
    write_df_to_db(df)
    verify_count()
    print("🎯 Загрузка zoo.arff завершена.")
