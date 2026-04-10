# Hospital ML — Previsão de Demanda no Pronto Socorro

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5-00A86B?style=flat-square)](https://lightgbm.readthedocs.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12%2B-4169E1?style=flat-square&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-MVP%20treinado-success?style=flat-square)](.)

> Pipeline de Machine Learning para previsão diária de volume de chegadas em um Pronto Socorro hospitalar, integrado a um Data Warehouse operacional baseado em PostgreSQL. Projeto em evolução, destinado a compor uma Central de ML dentro de uma plataforma de painéis hospitalares em produção.

---

## Visão geral

Este repositório contém os notebooks, scripts e artefatos de um projeto de Machine Learning aplicado à gestão hospitalar. O objetivo do primeiro modelo é **prever o volume diário de chegadas no Pronto Socorro** com horizonte de 1 a 7 dias, servindo de apoio ao dimensionamento de escala médica e de enfermagem, gestão de leitos e planejamento operacional.

O trabalho foi desenvolvido sobre um Data Warehouse PostgreSQL que recebe cargas incrementais diárias (via Apache Hop) de um sistema de informação hospitalar (HIS) baseado em Oracle. A plataforma de consumo é composta por painéis web modulares — este projeto adiciona um novo módulo: a **Central de Machine Learning**.

---

## Resultados do primeiro modelo

Modelo: **LightGBM** (gradient boosted decision trees) treinado em ~1000 dias de histórico (~237 mil registros de atendimentos).

| Métrica | Baseline (lag 7d) | LightGBM | Melhoria |
|---|---|---|---|
| **MAE (teste)** | 15.8 chegadas/dia | **14.1 chegadas/dia** | +10.9% |
| **MAPE (teste)** | 7.0% | **6.3%** | — |
| **RMSE (teste)** | — | 16.0 | — |

Com média de ~229 chegadas/dia no período de teste, o modelo apresenta **taxa de acerto de ~94%**. Métricas consistentes com o estado da arte em literatura de previsão de demanda de PS.

**Artefatos gerados:**
- `models/ps_volume_v1.pkl` (~250 KB) — modelo serializado
- `models/ps_volume_v1_meta.json` — metadados, métricas e hiperparâmetros

---

## Arquitetura do pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   HIS        │───▶│  Apache Hop  │───▶│  PostgreSQL  │
│  (Oracle)    │    │    (ETL)     │    │ (Data Wareh.)│
└──────────────┘    └──────────────┘    └──────┬───────┘
                                                │
                                        ┌───────┴────────┐
                                        │                │
                                        ▼                ▼
                              ┌──────────────┐  ┌──────────────┐
                              │  Notebooks   │  │ Open-Meteo   │
                              │   Jupyter    │◀─│  (clima)     │
                              │  (treino)    │  └──────────────┘
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │  Modelo .pkl │
                              │  + metadata  │
                              └──────┬───────┘
                                     │
                            ┌────────┴────────┐
                            │                 │
                            ▼                 ▼
                      ┌──────────┐      ┌──────────┐
                      │  Worker  │      │  Flask   │
                      │ (NSSM)   │─────▶│  API     │
                      └──────────┘      └────┬─────┘
                                              │
                                              ▼
                                        ┌──────────┐
                                        │ Frontend │
                                        │   (P30)  │
                                        └──────────┘
```

---

## Stack tecnológico

| Camada | Tecnologia |
|---|---|
| Fonte de dados | HIS proprietário (Oracle) |
| ETL | Apache Hop |
| Data Warehouse | PostgreSQL |
| Exploração e treino | Jupyter Lab + pandas + numpy |
| Visualização | matplotlib + seaborn |
| Séries temporais | statsmodels |
| Machine Learning | **LightGBM** + scikit-learn |
| Feriados | holidays (lib Python) |
| Clima histórico | [Open-Meteo API](https://open-meteo.com) (gratuita, sem API key) |
| Serialização | joblib |
| Integração (em desenvolvimento) | Flask + PostgreSQL + NSSM (Windows Service) |

---

## Estrutura do projeto

```
ml_workspace/
├── .env                              # credenciais do PostgreSQL (gitignored)
├── test_conexao.py                   # script de validação de ambiente
├── start_jupyter.bat                 # atalho para iniciar Jupyter Lab
├── notebooks/
│   ├── 01_visao_geral.ipynb          # inspeção inicial do dataset
│   ├── 02_analise_temporal.ipynb     # decomposição, feriados, clima, lags
│   └── 03_modelo_volume_ps.ipynb     # feature engineering, treino, avaliação
├── data/
│   ├── historico_chegadas.parquet    # dataset principal (~15 MB)
│   ├── clima_historico.parquet       # dados climáticos via Open-Meteo
│   └── serie_diaria_enriquecida.parquet
└── models/
    ├── ps_volume_v1.pkl              # modelo LightGBM serializado
    └── ps_volume_v1_meta.json        # metadados e métricas
```

---

## Metodologia

### 1. Extração e ETL

Query Oracle parametrizada une duas views do HIS (base + dados complementares) em um JOIN 1:1 por número de atendimento. Pipeline Apache Hop roda diariamente na madrugada, com janela de re-extração de 7 dias para capturar atualizações tardias (campos preenchidos ao longo do atendimento).

Padrão: **INSERT-only com deduplicação por PRIMARY KEY**, permitindo acumulação histórica sem perda de dados.

### 2. Análise exploratória

- **Decomposição sazonal** (statsmodels) confirmou forte sazonalidade semanal com amplitude de ~122 chegadas entre pico (segunda) e vale (domingo)
- Identificação de **surto epidêmico** em um período específico (detectado visualmente e tratado como feature binária `flag_surto`)
- **Desvio padrão do resíduo: 22.2** (apenas ~9% da média), indicando alta previsibilidade da série
- **Autocorrelação parcial** em lag 7 de 0.72, confirmando que "mesmo dia da semana passada" é o preditor mais forte

### 3. Engenharia de features

33 features agrupadas em 5 categorias:

- **Calendário (14)**: ano, mês, dia, dia da semana, codificação cíclica (sin/cos), trimestre, quinzena
- **Feriados (3)**: feriado nacional/estadual, véspera, pós-feriado
- **Evento especial (1)**: flag de surto epidêmico
- **Clima (8)**: temperatura (max/min/média), precipitação, chuva, umidade, vento — via Open-Meteo
- **Lags históricos (7)**: lag 1d, 2d, 7d, 14d + médias móveis 7d, 28d + volatilidade 7d

### 4. Split temporal

Divisão **cronológica** (nunca aleatória em séries temporais) em treino (90%), validação (6%) e teste (4%). O conjunto de teste representa o período mais recente e não é visto durante treino ou tuning.

### 5. Modelagem

LightGBM com hiperparâmetros conservadores e **early stopping** (para o treino quando a validação para de melhorar). Comparação contra baseline ingênuo (previsão = valor do mesmo dia da semana anterior) para validar o ganho do ML.

### 6. Avaliação

- **MAE** (erro médio absoluto): métrica principal, interpretável como "quantos pacientes o modelo erra em média"
- **MAPE** (erro percentual): facilita comunicação com gestão
- **Análise de resíduos**: erro por dia da semana e por tipo de dia (útil, fim de semana, feriado)
- **Calibração**: scatter realizado × previsto

---

## Features mais importantes

O modelo aprendeu o que era esperado: padrões temporais dominam o comportamento do PS.

| Rank | Feature | Importância |
|---|---|---|
| 1 | `lag_7d` | 45.3% |
| 2 | `lag_14d` | 12.5% |
| 3 | `dia_semana` | 11.8% |
| 4 | `lag_1d` | 6.7% |
| 5 | `media_movel_7d` | 6.3% |
| 6 | `dia_ano` | 2.4% |
| 7 | `lag_2d` | 2.0% |
| 8 | `is_feriado` | 1.9% |

As 5 primeiras features respondem por **82%** do poder preditivo do modelo — estrutura interpretável e auditável.

---

## Descobertas operacionais

Alguns achados do EDA que geraram valor além do modelo em si:

- **Feriados ≈ fim de semana**: feriados no PS comportam-se estatisticamente como domingos (média de 172 vs 171 chegadas/dia), ao contrário da intuição comum de que feriados lotam o PS por falta de atendimento em atenção básica
- **Pico absoluto da semana**: segunda-feira entre 9h e 11h é o horário mais crítico do PS
- **Segundo pico diário às 19h**: em todos os dias úteis, padrão consistente de "pós-expediente"
- **Domingo sem pico noturno**: o único dia da semana sem o segundo pico das 19h
- **Correlação com clima**: dias quentes e secos associados a aumento de chegadas; dias chuvosos reduzem a demanda (correlação fraca mas estatisticamente presente)

---

## Instalação e uso

### Pré-requisitos

- Python 3.11
- PostgreSQL 12+ com a tabela de histórico de chegadas populada via ETL
- Acesso de rede ao banco (ou execução local no servidor)

### Setup do ambiente

```bash
# Clonar repositório
git clone https://github.com/<usuario>/<repo>.git
cd <repo>

# Criar virtualenv
python -m venv venv_ml
venv_ml\Scripts\activate    # Windows
# source venv_ml/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

### Configuração

Criar arquivo `.env` na raiz:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=seu_banco
DB_USER=seu_usuario
DB_PASSWORD=sua_senha
```

### Execução dos notebooks

```bash
jupyter lab --notebook-dir=notebooks
```

Executar na ordem:
1. `01_visao_geral.ipynb`
2. `02_analise_temporal.ipynb`
3. `03_modelo_volume_ps.ipynb`

---

## Roadmap

### v1 — MVP (concluído)
- [x] Extração e ETL do histórico de chegadas
- [x] Análise exploratória e engenharia de features
- [x] Integração com dados climáticos
- [x] Treino do modelo LightGBM
- [x] Avaliação com MAE/MAPE
- [x] Serialização e metadados

### v1.1 — Deploy (em desenvolvimento)
- [ ] DDL das tabelas de registry e predições
- [ ] Worker de inferência diária (padrão NSSM)
- [ ] Backend Flask com endpoints de consulta
- [ ] Frontend Central de ML (painel P30)
- [ ] Monitoramento de drift e qualidade do modelo
- [ ] Alertas automáticos quando MAE exceder threshold

### v2 — Expansão
- [ ] Modelo de tempo de espera no PS
- [ ] Modelo de probabilidade de internação na chegada
- [ ] Modelo horário (granularidade intra-dia)
- [ ] Predição de readmissão em 30 dias
- [ ] Integração com alertas epidemiológicos externos
- [ ] Retreino automatizado semanal

---

## Princípios do projeto

- **Interpretabilidade acima de complexidade**: preferência por modelos tabulares auditáveis (LightGBM) em vez de deep learning para contextos clínicos
- **Validação temporal rigorosa**: zero data leakage, split cronológico, features sempre baseadas no passado
- **Baseline sempre presente**: qualquer modelo de ML deve superar a abordagem mais ingênua possível
- **Isolamento do ambiente**: desenvolvimento em virtualenv dedicado, sem contaminar o Python de produção
- **Responsabilidade clínica**: modelos atuam como apoio à decisão, nunca substituem o julgamento profissional

---

## Licença

Projeto interno de uso institucional. Código disponibilizado para fins de documentação, portfólio e referência metodológica.

---

## Autor

Desenvolvido como parte do setor de Tecnologia da Informação de uma instituição hospitalar, integrando-se a uma plataforma de painéis operacionais em produção.

Contribuições, sugestões e discussões são bem-vindas via Issues.