"""
Feature engineering compartilhado entre treino e inferência.
ATENÇÃO: este módulo é importado pelo worker E deve ser usado em qualquer
retreino futuro. Treino e inferência precisam usar o MESMO código.

Versão atual: compatível com modelo ps_volume v2.0
Mudança v1 → v2: adicionada flag_clinica_removida (Cardiologia removida em 20/01/2026)
"""
import numpy as np
import pandas as pd
import holidays
from datetime import timedelta


# =============================================================================
# CONSTANTES DE EVENTOS
# =============================================================================
# Primeiro dia em que a clinica esteve fora do fluxo do PS (21/01/2026)
DATA_REMOCAO_CARDIOLOGIA = '2026-01-21'

# Janela do surto de dengue de 2024
SURTO_DENGUE_INICIO = '2023-12-15'
SURTO_DENGUE_FIM = '2024-04-15'


# =============================================================================
# ORDEM DAS FEATURES (deve bater EXATAMENTE com o treino do modelo v2)
# =============================================================================

FEATURES_ORDER = [
    # Calendário
    'ano', 'mes', 'dia_mes', 'dia_semana', 'dia_ano', 'semana_ano', 'trimestre',
    'is_segunda', 'is_fim_semana', 'is_primeira_quinzena',
    'dia_semana_sin', 'dia_semana_cos', 'mes_sin', 'mes_cos',
    # Feriados
    'is_feriado', 'is_vespera_feriado', 'is_pos_feriado',
    # Eventos especiais
    'flag_surto_dengue_2024',
    'flag_clinica_removida',
    # Clima
    'temp_max', 'temp_min', 'temp_media',
    'precipitacao_mm', 'chuva_mm', 'horas_chuva',
    'umidade_media', 'vento_max',
    # Lags
    'lag_1d', 'lag_2d', 'lag_7d', 'lag_14d',
    'media_movel_7d', 'media_movel_28d', 'std_movel_7d',
]


def adicionar_features_calendario(df):
    """Adiciona todas as features de calendário ao DataFrame indexado por data."""
    df = df.copy()
    df['ano'] = df.index.year
    df['mes'] = df.index.month
    df['dia_mes'] = df.index.day
    df['dia_semana'] = df.index.dayofweek
    df['dia_ano'] = df.index.dayofyear
    df['semana_ano'] = df.index.isocalendar().week.astype(int).values
    df['trimestre'] = df.index.quarter
    df['is_segunda'] = (df['dia_semana'] == 0).astype(int)
    df['is_fim_semana'] = (df['dia_semana'] >= 5).astype(int)
    df['is_primeira_quinzena'] = (df['dia_mes'] <= 15).astype(int)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    return df


def adicionar_features_feriados(df):
    """Adiciona flags de feriado, véspera e pós-feriado."""
    df = df.copy()
    anos = list(range(df.index.year.min(), df.index.year.max() + 1))
    feriados_br = holidays.Brazil(years=anos, subdiv='DF')
    feriado_datas = set(feriados_br.keys())

    df['is_feriado'] = df.index.to_series().apply(
        lambda d: d.date() in feriados_br
    ).astype(int)
    df['is_vespera_feriado'] = df.index.to_series().apply(
        lambda d: (d.date() + timedelta(days=1)) in feriado_datas
    ).astype(int)
    df['is_pos_feriado'] = df.index.to_series().apply(
        lambda d: (d.date() - timedelta(days=1)) in feriado_datas
    ).astype(int)
    return df


def adicionar_flag_surto(df):
    """Marca o período do surto de dengue 2024."""
    df = df.copy()
    df['flag_surto_dengue_2024'] = (
        (df.index >= SURTO_DENGUE_INICIO) & (df.index <= SURTO_DENGUE_FIM)
    ).astype(int)
    return df


def adicionar_flag_clinica_removida(df):
    """
    Marca dias a partir da remoção da Cardiologia do fluxo do PS.
    0 = antes da remoção (Cardiologia no fluxo)
    1 = a partir de DATA_REMOCAO_CARDIOLOGIA (Cardiologia fora do fluxo)
    """
    df = df.copy()
    df['flag_clinica_removida'] = (
        df.index >= DATA_REMOCAO_CARDIOLOGIA
    ).astype(int)
    return df


def adicionar_lags(df, coluna_target='chegadas'):
    """
    Adiciona lags e médias móveis.
    IMPORTANTE: usa shift(1) em tudo para garantir que só vê o passado.
    """
    df = df.copy()
    df['lag_1d'] = df[coluna_target].shift(1)
    df['lag_2d'] = df[coluna_target].shift(2)
    df['lag_7d'] = df[coluna_target].shift(7)
    df['lag_14d'] = df[coluna_target].shift(14)
    df['media_movel_7d'] = df[coluna_target].shift(1).rolling(7).mean()
    df['media_movel_28d'] = df[coluna_target].shift(1).rolling(28).mean()
    df['std_movel_7d'] = df[coluna_target].shift(1).rolling(7).std()
    return df


def construir_features_completas(df_chegadas, df_clima):
    """
    Pipeline completo de feature engineering.

    Parâmetros:
        df_chegadas: DataFrame indexado por data com coluna 'chegadas'
        df_clima:    DataFrame indexado por data com colunas climáticas

    Retorna:
        DataFrame com todas as 34 features na ordem correta (FEATURES_ORDER).
    """
    df = df_chegadas.copy()
    df = adicionar_features_calendario(df)
    df = adicionar_features_feriados(df)
    df = adicionar_flag_surto(df)
    df = adicionar_flag_clinica_removida(df)
    df = adicionar_lags(df)
    df = df.join(df_clima, how='left')
    return df