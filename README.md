# 🌟 Mirach - Plataforma de Previsão de Séries Temporais

**Mirach** é uma plataforma SaaS completa para previsão de séries temporais, desenvolvida em Django puro. O nome é uma referência à estrela Mirach (β Andromedae), simbolizando a orientação e precisão nas previsões.

## 🚀 Características Principais

### 📊 Algoritmos de Previsão Disponíveis
- **ARIMA** - Modelo autorregressivo integrado de médias móveis
- **ETS** - Suavização exponencial (Exponential Smoothing)
- **Prophet** - Algoritmo desenvolvido pelo Facebook
- **LSTM** - Redes neurais recorrentes
- **Regressão Linear** - Modelo linear clássico
- **Regressão Polinomial** - Modelos não-lineares
- **Ridge/Lasso** - Regressão com regularização
- **Random Forest** - Ensemble de árvores de decisão
- **XGBoost** - Gradient boosting otimizado
- **Support Vector Regression** - Máquinas de vetores de suporte
- **Redes Neurais** - Perceptrons multicamadas

### 🎯 Funcionalidades Core
- **Gestão de Projetos**: Organize seus projetos de previsão
- **Upload de Datasets**: Suporte completo para planilhas Excel
- **Mapeamento Inteligente**: Sistema automático de mapeamento de colunas
- **Visualizações Interativas**: Gráficos dinâmicos com Plotly
- **Dashboard Completo**: Acompanhe métricas e resultados
- **Sistema de Usuários**: Autenticação e perfis personalizados

## 🛠️ Tecnologias Utilizadas

### Backend
- **Django 4.2.6** - Framework web principal
- **Django REST Framework** - API RESTful
- **PostgreSQL/SQLite** - Banco de dados
- **Celery** - Processamento assíncrono (futuro)

### Data Science
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **SciPy** - Computação científica
- **Scikit-learn** - Machine learning
- **Statsmodels** - Modelos estatísticos
- **OpenPyXL** - Processamento de Excel

### Frontend
- **Bootstrap 5** - Framework CSS
- **Font Awesome** - Ícones
- **Plotly** - Visualizações interativas
- **JavaScript** - Interatividade

## 📋 Pré-requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)
- Git

## 🚀 Instalação e Configuração

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd mirach
```

### 2. Crie e ative o ambiente virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure o banco de dados
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Crie um superusuário
```bash
python manage.py createsuperuser
```

### 6. Execute o servidor de desenvolvimento
```bash
python manage.py runserver
```

Acesse `http://127.0.0.1:8000` no seu navegador.

## 📁 Estrutura do Projeto

```
mirach/
├── accounts/          # Gestão de usuários e autenticação
├── projects/          # Gestão de projetos
├── datasets/          # Upload e processamento de dados
├── predictions/       # Algoritmos e execução de previsões
├── templates/         # Templates HTML
├── static/           # Arquivos estáticos (CSS, JS, imagens)
├── media/            # Arquivos de upload
├── setup/            # Configurações do Django
└── requirements.txt  # Dependências do projeto
```

## 🎯 Como Usar

### 1. Criar um Projeto
- Faça login na plataforma
- Clique em "Novo Projeto"
- Defina nome e descrição

### 2. Upload de Dataset
- Acesse "Upload Dataset"
- Selecione seu projeto
- Faça upload de uma planilha Excel
- Configure o mapeamento de colunas

### 3. Executar Previsões
- Acesse "Nova Previsão"
- Selecione projeto, dataset e algoritmo
- Configure parâmetros específicos
- Execute a previsão

### 4. Visualizar Resultados
- Acesse o dashboard de resultados
- Visualize gráficos interativos
- Analise métricas de performance
- Exporte resultados

## 🔧 Configurações Avançadas

### Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto:

```env
DEBUG=True
SECRET_KEY=sua-chave-secreta-aqui
DATABASE_URL=sqlite:///db.sqlite3
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Configuração de Email
Para notificações por email, configure no `settings.py`:

```python
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'seu-email@gmail.com'
EMAIL_HOST_PASSWORD = 'sua-senha-de-app'
```

## 📊 Modelos de Dados

### Projeto
- Nome, descrição, proprietário
- Datas de criação e atualização
- Status ativo/inativo

### Dataset
- Arquivo original (Excel)
- Metadados (linhas, colunas, tipos)
- Status de processamento
- Mapeamento de colunas

### Previsão
- Algoritmo utilizado
- Parâmetros de configuração
- Métricas de performance
- Resultados e visualizações

## 🚀 Roadmap

### Versão 1.1
- [ ] Integração com Prophet
- [ ] Suporte a mais formatos de arquivo
- [ ] API REST completa
- [ ] Documentação da API

### Versão 1.2
- [ ] Processamento assíncrono com Celery
- [ ] Notificações por email
- [ ] Exportação de relatórios
- [ ] Dashboard avançado

### Versão 2.0
- [ ] Machine Learning automático (AutoML)
- [ ] Integração com APIs externas
- [ ] Colaboração em equipe
- [ ] Versionamento de modelos

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Suporte

Para suporte e dúvidas:
- Abra uma [issue](https://github.com/seu-usuario/mirach/issues)
- Entre em contato: [seu-email@exemplo.com]

## 🙏 Agradecimentos

- Django Software Foundation
- Comunidade Python
- Contribuidores do projeto

---

**Mirach** - Orientando o futuro através da previsão de séries temporais ⭐