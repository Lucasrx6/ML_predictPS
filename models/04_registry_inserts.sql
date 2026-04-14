-- INSERT dos 4 modelos no ml_modelos_registry
-- Gerado automaticamente pelo Notebook 02


INSERT INTO public.ml_modelos_registry (
    nome_modelo, versao, descricao, categoria, algoritmo,
    caminho_pkl, caminho_metadata,
    periodo_treino_inicio, periodo_treino_fim, num_amostras_treino,
    mae_teste, mape_teste, rmse_teste,
    num_features, status, criado_por, observacoes
) VALUES (
    'fat_internado', 'v1.0',
    'Internado - faturamento diario, log1p=True',
    'previsao_financeira', 'LightGBM',
    'C:/ml_workspace/models/fat_internado_v1.pkl',
    'C:/ml_workspace/models/fat_internado_v1_meta.json',
    '2024-01-01', '2026-02-28',
    776,
    113171.07, 742.49, 170599.53,
    42, 'producao', 'lucas',
    'v1 inicial. MAE_baseline R$ 126980.46, ganho 10.9%.'
);

INSERT INTO public.ml_modelos_registry (
    nome_modelo, versao, descricao, categoria, algoritmo,
    caminho_pkl, caminho_metadata,
    periodo_treino_inicio, periodo_treino_fim, num_amostras_treino,
    mae_teste, mape_teste, rmse_teste,
    num_features, status, criado_por, observacoes
) VALUES (
    'fat_ps', 'v1.0',
    'Pronto socorro - faturamento diario, log1p=True',
    'previsao_financeira', 'LightGBM',
    'C:/ml_workspace/models/fat_ps_v1.pkl',
    'C:/ml_workspace/models/fat_ps_v1_meta.json',
    '2024-01-01', '2026-02-28',
    762,
    13417.40, 38.27, 23134.50,
    37, 'producao', 'lucas',
    'v1 inicial. MAE_baseline R$ 21665.00, ganho 38.1%.'
);

INSERT INTO public.ml_modelos_registry (
    nome_modelo, versao, descricao, categoria, algoritmo,
    caminho_pkl, caminho_metadata,
    periodo_treino_inicio, periodo_treino_fim, num_amostras_treino,
    mae_teste, mape_teste, rmse_teste,
    num_features, status, criado_por, observacoes
) VALUES (
    'fat_ambulatorial', 'v1.0',
    'Atendimento Ambulatorial - faturamento diario, log1p=True',
    'previsao_financeira', 'LightGBM',
    'C:/ml_workspace/models/fat_ambulatorial_v1.pkl',
    'C:/ml_workspace/models/fat_ambulatorial_v1_meta.json',
    '2024-01-01', '2026-02-28',
    637,
    10432.69, 217.51, 13982.55,
    30, 'producao', 'lucas',
    'v1 inicial. MAE_baseline R$ 10764.15, ganho 3.1%.'
);

INSERT INTO public.ml_modelos_registry (
    nome_modelo, versao, descricao, categoria, algoritmo,
    caminho_pkl, caminho_metadata,
    periodo_treino_inicio, periodo_treino_fim, num_amostras_treino,
    mae_teste, mape_teste, rmse_teste,
    num_features, status, criado_por, observacoes
) VALUES (
    'fat_externo', 'v1.0',
    'Externo - faturamento diario, log1p=False',
    'previsao_financeira', 'LightGBM',
    'C:/ml_workspace/models/fat_externo_v1.pkl',
    'C:/ml_workspace/models/fat_externo_v1_meta.json',
    '2024-01-01', '2026-02-28',
    776,
    5735.70, 102.82, 7418.40,
    30, 'producao', 'lucas',
    'v1 inicial. MAE_baseline R$ 8502.37, ganho 32.5%.'
);
