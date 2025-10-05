from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()  # читает .env

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_SCHEMA = os.getenv("DB_SCHEMA", "ml_data")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args={"options": f"-csearch_path={DB_SCHEMA}"}
)

def test_connection():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT current_schema();"))
        print("Connected to schema:", res.scalar())

if __name__ == "__main__":
    test_connection()