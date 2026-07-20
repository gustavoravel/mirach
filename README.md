# Mirach — Plataforma de Previsão de Séries Temporais

**Mirach** é uma plataforma SaaS para previsão de séries temporais, desenvolvida em Django. O nome referencia a estrela Mirach (β Andromedae), simbolizando orientação e precisão nas previsões.

## Características

### Algoritmos
- **ARIMA / SARIMAX** — com suporte a regressores exógenos e `pmdarima.auto_arima`
- **ETS** — suavização exponencial
- **Prophet** — com regressors opcionais
- **LightGBM / XGBoost / Random Forest** — ML com lags, rolling e features exógenas
- **Ridge / Lasso / Linear / SVR / MLP**
- **Baselines** — naive e seasonal naive (usados no campeonato)
- **LSTM** — disponível se TensorFlow estiver instalado

### Robustez do motor
- Previsões **reproduzíveis** (seeds fixos; sem ruído artificial)
- Uso real das **features mapeadas** como exógenas
- Pipeline de processamento com perfil de dados (`data_profile`), frequência inferida e gaps
- Métricas honestas (sem zeros falsos quando não há validação)
- Intervalos de confiança quando o modelo permite
- **Campeonato empírico** (walk-forward) para recomendar o melhor algoritmo

### Camada agêntica (NVIDIA NIM)
- Modelo padrão: `nvidia/nemotron-3-ultra-550b-a55b`
- **IngestAgent** — sugere mapeamento de colunas a partir do perfil do dataset
- **OrchestratorAgent** — modo Auto no wizard (tools + campeonato)
- **NarrativeAgent** — insights em PT-BR ancorados em métricas reais
- O LLM **nunca** inventa valores de forecast; só interpreta e explica
- Fallback determinístico se a API NIM estiver indisponível

### Produto
- Projetos, memberships, planos e webhooks
- Upload Excel/CSV, mapeamento de colunas, Celery para jobs
- Visualizações Plotly, export CSV/JSON, API REST

## Stack

| Camada | Tecnologia |
|--------|------------|
| Backend | Django 4.2, DRF, Celery, Redis |
| DB / Storage | PostgreSQL, MinIO/S3 |
| DS | pandas, numpy, scikit-learn, statsmodels, prophet, xgboost, lightgbm, pmdarima |
| IA | OpenAI SDK → NVIDIA NIM (`openai`, `pydantic`) |
| Frontend | Bootstrap 5, Plotly |

## Pré-requisitos

- Python 3.10+ (ou Docker / Docker Compose)
- Redis (para Celery)
- Chave NVIDIA NIM (opcional, mas necessária para a camada agêntica)

## Instalação

### Com Docker (recomendado)

```bash
cp .env.example .env   # se existir; ou edite .env
# Defina NVIDIA_NIM_API_KEY no .env
docker compose up --build
docker compose exec web python manage.py migrate
docker compose exec web python manage.py init_models
docker compose exec web python manage.py createsuperuser
```

Acesse `http://127.0.0.1:8000`.

### Local (venv)

```bash
git clone <url-do-repositorio>
cd mirach
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py init_models
python manage.py createsuperuser
# Em outro terminal: celery -A setup worker -l info
python manage.py runserver
```

## Variáveis de ambiente

Trecho relevante do `.env`:

```env
DEBUG=True
SECRET_KEY=sua-chave-secreta
DATABASE_URL=postgres://mirach:mirach@db:5432/mirach
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# NVIDIA NIM (camada agêntica)
NVIDIA_NIM_API_KEY=nvapi-...
NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_NIM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

Sem `NVIDIA_NIM_API_KEY`, ingest/orquestração/narrativa usam fallbacks heurísticos/templates — o forecast clássico continua funcionando.

## Estrutura

```
mirach/
├── accounts/              # Auth, planos, billing, API tokens
├── projects/              # Projetos, memberships, webhooks
├── datasets/              # Upload, mapeamento, perfil, pipeline
├── predictions/
│   ├── algorithms.py      # Forecasters
│   ├── championship.py    # Campeonato walk-forward
│   ├── services.py        # Orquestração Django
│   ├── tasks.py           # Celery (previsão, auto-plan, championship)
│   └── llm/               # Cliente NIM + agentes
├── setup/                 # settings, celery, urls
├── templates/ / static/
├── docker-compose.yml
└── requirements.txt
```

## Como usar

1. **Projeto** — crie um projeto e (opcional) convide membros.
2. **Dataset** — faça upload CSV/Excel → mapeie colunas (sugestão automática / IA) → processe.
3. **Previsão**
   - Manual: escolha o algoritmo no wizard, ou
   - **Modo Auto**: campeonato + OrchestratorAgent (NIM) escolhem o plano.
4. **Executar** — job Celery treina, prevê, gera CIs/métricas e insights IA.
5. **Explorar** — gráficos, métricas, `explainability.ai_insights`, export.

### APIs úteis

| Endpoint | Descrição |
|----------|-----------|
| `GET /predictions/api/recommendations/<dataset_id>/` | Ranking empírico do campeonato |
| `GET /predictions/api/auto-plan/<dataset_id>/` | ModelPlan (sync; `?async=1` enfileira Celery) |
| `GET /predictions/api/backtest/<dataset_id>/` | Backtest rolling-origin |
| `GET /predictions/api/visualization/<pk>/` | Dados para gráficos |

## Modelos de dados (resumo)

- **Dataset** — arquivo, `data_profile`, `ai_interpretation`, mapeamentos
- **Prediction** — algoritmo, parâmetros, métricas, `predictions_data`, `explainability` (inclui `ai_insights`)
- **PredictionResult** — ponto a ponto + intervalo de confiança

## Roadmap

### Feito nesta linha
- [x] Celery + Redis
- [x] Features exógenas nos modelos
- [x] Campeonato / AutoML leve (walk-forward)
- [x] Camada agêntica NVIDIA NIM (ingest, orquestração, narrativa)

### Próximos passos
- [ ] Tuning de hiperparâmetros com Optuna (budget por plano)
- [ ] Cache Redis mais amplo de perfis LLM
- [ ] Documentação OpenAPI completa (drf-spectacular)
- [ ] Modelos de fundação de séries (TimeGPT / Chronos) como opção premium

## Contribuição

1. Fork → branch `feat/...` ou `fix/...`
2. Commits no estilo Conventional Commits (`feat`, `fix`, `chore`, …)
3. MR/PR pequeno e focado; referencie issues com `#N` quando houver

## Licença

MIT — veja [LICENSE](LICENSE) se presente no repositório.

---

**Mirach** — orientando o futuro através da previsão de séries temporais.
