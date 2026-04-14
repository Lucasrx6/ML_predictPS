"""
========================================
WORKER ML — PREVISÃO DE VOLUME DO PS
========================================

Roda diariamente após o ETL principal.
Gera predições para os próximos 7 dias e atualiza valores realizados.

Fluxo:
  1. Carrega modelo ps_volume_v2 (LightGBM, 34 features)
  2. Lê histórico de chegadas do PostgreSQL (exclui Cardiologia automaticamente
     se a tabela já estiver limpa; caso contrário, exclusão fica no notebook de
     treino — em produção o dado já vem sem Cardiologia a partir de 21/01/2026)
  3. Busca clima histórico + forecast via Open-Meteo
  4. Computa features para os próximos 7 dias (inclui flag_clinica_removida)
  5. Gera predições e grava em ml_ps_predicoes (UPSERT)
  6. Atualiza valor_realizado das predições antigas

Uso:
  python ml_ps_volume_worker.py

Logs em: ../logs/ml_ps_volume.log
"""
import os
import sys
import json
import hashlib
import logging
from datetime import datetime, timedelta, date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

# Adiciona o diretório atual ao path para importar modules locais
sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import get_db_connection, get_dict_cursor, get_sqlalchemy_engine
from features import construir_features_completas, FEATURES_ORDER


# ========================================
# CONFIGURAÇÃO
# ========================================

BASE_DIR = Path(__file__).resolve().parent.parent  # ml_workspace/
MODEL_PATH = BASE_DIR / 'models' / 'ps_volume_v2.pkl'
META_PATH = BASE_DIR / 'models' / 'ps_volume_v2_meta.json'
LOG_PATH = BASE_DIR / 'logs' / 'ml_ps_volume.log'

# Coordenadas para a API de clima
LAT = -15.82
LON = -48.11

# Configuração do modelo
MODELO_NOME = 'ps_volume'
MODELO_VERSAO = 'v2.0'
HORIZONTE_DIAS = 7

# Logging
LOG_PATH.parent.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ========================================
# FUNÇÕES AUXILIARES
# ========================================

def carregar_modelo():
    """Carrega o modelo .pkl e os metadados."""
    logger.info(f"Carregando modelo: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo do modelo nao encontrado: {MODEL_PATH}. "
            f"Certifique-se de rodar o Notebook 04 antes de iniciar o worker."
        )
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    logger.info(f"Modelo carregado: {meta['nome']} ({meta['versao']})")
    return model, meta


def buscar_modelo_id(conn, nome, versao):
    """Busca o id do modelo no registry."""
    cursor = get_dict_cursor(conn)
    cursor.execute("""
        SELECT id FROM public.ml_modelos_registry
        WHERE nome_modelo = %s AND versao = %s
    """, (nome, versao))
    row = cursor.fetchone()
    cursor.close()
    if row is None:
        raise RuntimeError(
            f"Modelo {nome} {versao} nao encontrado no registry. "
            f"Execute o INSERT do v2 em ml_modelos_registry antes do deploy."
        )
    return row['id']


def buscar_historico_chegadas(conn):
    """
    Lê o histórico completo de chegadas agregado por dia.
    IMPORTANTE: exclui explicitamente Cardiologia do total, garantindo
    consistência com o treino do modelo v2.
    """
    logger.info("Buscando histórico de chegadas no PostgreSQL (excluindo Cardiologia)...")
    query = """
        SELECT
            DATE(dt_entrada) AS data,
            COUNT(*)::int AS chegadas
        FROM public.ml_ps_historico_chegadas
        WHERE dt_entrada < CURRENT_DATE
          AND (ds_clinica IS NULL OR ds_clinica <> 'Cardiologia')
        GROUP BY DATE(dt_entrada)
        ORDER BY data
    """
    engine = get_sqlalchemy_engine()
    df = pd.read_sql(query, engine)
    df['data'] = pd.to_datetime(df['data'])
    df = df.set_index('data').asfreq('D')
    df['chegadas'] = df['chegadas'].fillna(0).astype('float64')
    logger.info(f"Histórico carregado: {len(df)} dias ({df.index.min().date()} a {df.index.max().date()})")
    return df


def buscar_clima_completo(data_inicio, data_fim):
    """
    Busca clima combinando duas APIs da Open-Meteo:
      - /v1/archive  → histórico oficial (qualquer data passada)
      - /v1/forecast → previsão (até 16 dias no futuro)

    Retorna um DataFrame único com toda a janela coberta.
    """
    hoje = pd.Timestamp(date.today())
    daily_vars = [
        'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
        'precipitation_sum', 'rain_sum', 'precipitation_hours',
        'wind_speed_10m_max', 'relative_humidity_2m_mean',
    ]

    rename_map = {
        'temperature_2m_max': 'temp_max',
        'temperature_2m_min': 'temp_min',
        'temperature_2m_mean': 'temp_media',
        'precipitation_sum': 'precipitacao_mm',
        'rain_sum': 'chuva_mm',
        'precipitation_hours': 'horas_chuva',
        'wind_speed_10m_max': 'vento_max',
        'relative_humidity_2m_mean': 'umidade_media',
    }

    frames = []

    # ----- HISTÓRICO (/v1/archive) -----
    fim_hist = (hoje - timedelta(days=1)).date()
    if pd.Timestamp(data_inicio) <= pd.Timestamp(fim_hist):
        logger.info(f"Buscando clima HISTÓRICO de {data_inicio} a {fim_hist}...")
        url_hist = 'https://archive-api.open-meteo.com/v1/archive'
        params_hist = {
            'latitude': LAT,
            'longitude': LON,
            'start_date': pd.Timestamp(data_inicio).strftime('%Y-%m-%d'),
            'end_date': fim_hist.strftime('%Y-%m-%d'),
            'daily': ','.join(daily_vars),
            'timezone': 'America/Sao_Paulo',
        }
        r = requests.get(url_hist, params=params_hist, timeout=60)
        r.raise_for_status()
        df_hist = pd.DataFrame(r.json()['daily'])
        frames.append(df_hist)
        logger.info(f"  Histórico: {len(df_hist)} dias recebidos")

    # ----- FORECAST (/v1/forecast) -----
    if pd.Timestamp(data_fim) >= hoje:
        logger.info(f"Buscando clima FORECAST de {hoje.date()} a {data_fim}...")
        url_fc = 'https://api.open-meteo.com/v1/forecast'
        params_fc = {
            'latitude': LAT,
            'longitude': LON,
            'start_date': hoje.strftime('%Y-%m-%d'),
            'end_date': pd.Timestamp(data_fim).strftime('%Y-%m-%d'),
            'daily': ','.join(daily_vars),
            'timezone': 'America/Sao_Paulo',
        }
        r = requests.get(url_fc, params=params_fc, timeout=60)
        r.raise_for_status()
        df_fc = pd.DataFrame(r.json()['daily'])
        frames.append(df_fc)
        logger.info(f"  Forecast: {len(df_fc)} dias recebidos")

    clima = pd.concat(frames, ignore_index=True)
    clima['time'] = pd.to_datetime(clima['time'])
    clima = clima.set_index('time')
    clima.index.name = 'data'
    clima = clima.rename(columns=rename_map)
    clima = clima[~clima.index.duplicated(keep='first')]

    logger.info(f"Clima total combinado: {len(clima)} dias ({clima.index.min().date()} a {clima.index.max().date()})")
    return clima


def gerar_predicoes(model, df_features, mae_modelo):
    """Gera predições para os dias futuros do dataframe."""
    hoje = pd.Timestamp(date.today())
    df_futuro = df_features[df_features.index >= hoje].head(HORIZONTE_DIAS)

    if len(df_futuro) == 0:
        logger.warning("Nenhum dia futuro disponível para predição")
        return []

    X = df_futuro[FEATURES_ORDER]

    # Valida que temos as 34 features do v2
    n_esperadas = len(FEATURES_ORDER)
    if X.shape[1] != n_esperadas:
        raise RuntimeError(
            f"Numero de features incorreto: esperado {n_esperadas}, recebido {X.shape[1]}. "
            f"Verifique FEATURES_ORDER em features.py"
        )

    predicoes = model.predict(X)

    resultados = []
    for i, (data, pred) in enumerate(zip(df_futuro.index, predicoes)):
        features_raw = X.loc[data].to_dict()
        features_dict = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in features_raw.items()
        }
        features_json = json.dumps(features_dict, default=str, sort_keys=True)
        hash_feat = hashlib.md5(features_json.encode()).hexdigest()

        resultados.append({
            'dt_alvo': data.date(),
            'horizonte_dias': i + 1,
            'valor_previsto': round(float(pred), 2),
            'intervalo_inferior': round(float(pred - mae_modelo), 2),
            'intervalo_superior': round(float(pred + mae_modelo), 2),
            'features_usadas': features_dict,
            'hash_features': hash_feat,
        })

    logger.info(f"Geradas {len(resultados)} predições para os próximos {HORIZONTE_DIAS} dias")
    return resultados


def gravar_predicoes(conn, modelo_id, modelo_nome, modelo_versao, predicoes):
    """Insere/atualiza predições em ml_ps_predicoes (upsert)."""
    cursor = conn.cursor()
    for pred in predicoes:
        features_dict_safe = {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in pred['features_usadas'].items()
        }
        cursor.execute("""
            INSERT INTO public.ml_ps_predicoes (
                dt_alvo, horizonte_dias,
                valor_previsto, intervalo_inferior, intervalo_superior,
                modelo_id, modelo_nome, modelo_versao,
                features_usadas, hash_features
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (dt_alvo, horizonte_dias, modelo_id) DO UPDATE SET
                dt_geracao = NOW(),
                valor_previsto = EXCLUDED.valor_previsto,
                intervalo_inferior = EXCLUDED.intervalo_inferior,
                intervalo_superior = EXCLUDED.intervalo_superior,
                features_usadas = EXCLUDED.features_usadas,
                hash_features = EXCLUDED.hash_features,
                valor_realizado = NULL,
                erro_absoluto = NULL,
                erro_percentual = NULL,
                dt_atualizacao_real = NULL
        """, (
            pred['dt_alvo'], pred['horizonte_dias'],
            pred['valor_previsto'], pred['intervalo_inferior'], pred['intervalo_superior'],
            modelo_id, modelo_nome, modelo_versao,
            json.dumps(features_dict_safe, default=str),
            pred['hash_features'],
        ))
    conn.commit()
    cursor.close()
    logger.info(f"{len(predicoes)} predições gravadas/atualizadas")


def atualizar_valores_realizados(conn):
    """
    Para cada predição com dt_alvo no passado e valor_realizado NULL,
    busca o valor real na ml_ps_historico_chegadas (sem Cardiologia) e atualiza.
    """
    logger.info("Atualizando valores realizados das predições antigas...")
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE public.ml_ps_predicoes p
        SET
            valor_realizado = sub.real,
            erro_absoluto   = ABS(p.valor_previsto - sub.real),
            erro_percentual = CASE
                WHEN sub.real >= 10
                THEN LEAST(
                    ABS(p.valor_previsto - sub.real) / sub.real * 100,
                    9999.999
                )
                ELSE NULL
            END,
            dt_atualizacao_real = NOW()
        FROM (
            SELECT DATE(dt_entrada) AS data, COUNT(*)::numeric AS real
            FROM public.ml_ps_historico_chegadas
            WHERE dt_entrada < CURRENT_DATE
              AND (ds_clinica IS NULL OR ds_clinica <> 'Cardiologia')
            GROUP BY DATE(dt_entrada)
        ) sub
        WHERE p.dt_alvo = sub.data
          AND p.valor_realizado IS NULL
          AND p.dt_alvo < CURRENT_DATE
    """)
    atualizadas = cursor.rowcount
    conn.commit()
    cursor.close()
    logger.info(f"{atualizadas} predições atualizadas com valor realizado")


# ========================================
# MAIN
# ========================================

def main():
    inicio = datetime.now()
    logger.info("=" * 60)
    logger.info(f"WORKER ML PS VOLUME {MODELO_VERSAO} — INÍCIO")
    logger.info("=" * 60)

    try:
        # 1. Carrega modelo
        model, meta = carregar_modelo()
        mae_modelo = meta['metricas']['mae_teste']

        # 2. Conecta no banco
        conn = get_db_connection()
        modelo_id = buscar_modelo_id(conn, MODELO_NOME, MODELO_VERSAO)
        logger.info(f"Modelo registry id: {modelo_id}")

        # 3. Lê histórico (sem Cardiologia)
        df_historico = buscar_historico_chegadas(conn)

        # 4. Estende com dias futuros (NaN nas chegadas)
        ultima_data = df_historico.index.max()
        datas_futuras = pd.date_range(
            ultima_data + timedelta(days=1),
            periods=HORIZONTE_DIAS,
            freq='D'
        )
        df_futuro = pd.DataFrame(
            {'chegadas': np.nan},
            index=datas_futuras,
            dtype='float64'
        )
        df_completo = pd.concat([df_historico, df_futuro])
        df_completo['chegadas'] = pd.to_numeric(df_completo['chegadas'], errors='coerce')

        # 5. Busca clima
        clima = buscar_clima_completo(
            df_completo.index.min().date(),
            df_completo.index.max().date()
        )

        # 6. Feature engineering (inclui flag_clinica_removida)
        df_features = construir_features_completas(df_completo, clima)

        # 7. Gera e grava predições
        predicoes = gerar_predicoes(model, df_features, mae_modelo)
        if predicoes:
            gravar_predicoes(conn, modelo_id, MODELO_NOME, MODELO_VERSAO, predicoes)

        # 8. Fecha o ciclo (atualiza valores realizados)
        atualizar_valores_realizados(conn)

        conn.close()

        duracao = (datetime.now() - inicio).total_seconds()
        logger.info("=" * 60)
        logger.info(f"WORKER FINALIZADO — duração: {duracao:.1f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"ERRO FATAL: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()