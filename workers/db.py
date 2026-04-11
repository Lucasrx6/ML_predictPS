"""
Helpers de conexão com PostgreSQL.
Usa o mesmo .env do ml_workspace.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine

# Sobe um nível pra encontrar o .env na raiz do ml_workspace
ENV_PATH = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(ENV_PATH)


def get_db_connection():
    """Retorna uma conexão psycopg2 ao PostgreSQL."""
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )


def get_dict_cursor(conn):
    """Retorna um cursor com RealDictCursor (mesmo padrão dos outros painéis)."""
    return conn.cursor(cursor_factory=RealDictCursor)




def get_sqlalchemy_engine():
    """Retorna um engine SQLAlchemy (usado quando o pandas precisa)."""
    return create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )