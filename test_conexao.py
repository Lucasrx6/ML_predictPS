"""
Teste de conexão com PostgreSQL e validação da tabela ml_ps_historico_chegadas.
Roda isso antes de abrir o Jupyter pra garantir que tudo está plugado.
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

print("=" * 60)
print("TESTE DE CONEXÃO - ML Workspace")
print("=" * 60)
print(f"Host: {DB_HOST}:{DB_PORT}")
print(f"Database: {DB_NAME}")
print(f"User: {DB_USER}")
print()

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    print("[OK] Conexão estabelecida com sucesso")

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Valida que a tabela existe e conta linhas
    cursor.execute("""
        SELECT COUNT(*) AS total_linhas,
               MIN(dt_entrada) AS primeira_entrada,
               MAX(dt_entrada) AS ultima_entrada
        FROM public.ml_ps_historico_chegadas
    """)
    resultado = cursor.fetchone()

    print()
    print("=" * 60)
    print("TABELA: ml_ps_historico_chegadas")
    print("=" * 60)
    print(f"Total de linhas:   {resultado['total_linhas']:,}")
    print(f"Primeira entrada:  {resultado['primeira_entrada']}")
    print(f"Última entrada:    {resultado['ultima_entrada']}")
    print()
    print("[OK] Tudo pronto para o Jupyter")

    cursor.close()
    conn.close()

except Exception as e:
    print()
    print(f"[ERRO] {type(e).__name__}: {e}")
    print()
    print("Verifique:")
    print("  1. O arquivo .env está com as credenciais corretas")
    print("  2. O PostgreSQL está rodando (netstat -an | findstr 5432)")
    print("  3. A tabela ml_ps_historico_chegadas existe")