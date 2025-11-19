# 🤖 Mitra AI - Emotionally Intelligent AI Assistant

Mitra is an advanced AI assistant that combines deep reasoning capabilities with emotional intelligence to provide thoughtful, supportive interactions through Telegram.

## 🌟 Features

### Core Capabilities
- **Deep Reasoning**: Multi-step problem solving with chain-of-thought processing
- **Emotional Intelligence**: Sentiment analysis and emotion detection to adapt responses
- **Crisis Detection**: Identifies crisis situations and provides appropriate resources
- **Safety Features**: Content moderation and safety boundaries
- **Conversation Memory**: Maintains context across conversation history
- **Rate Limiting**: Built-in abuse prevention and rate limiting

### Technical Features
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Structured Logging**: Comprehensive logging with correlation IDs
- **Error Handling**: Robust error handling with user-friendly messages
- **Azure Integration**: Designed for Azure OpenAI and Azure Container Apps
- **CI/CD Pipeline**: Automated testing, building, and deployment via GitHub Actions
- **Type Safety**: Full type hints with mypy checking
- **Containerized**: Docker support for consistent deployments

## 🏗️ Architecture

```
mitra/
├── core/               # Core AI intelligence engine
│   ├── engine.py      # Main AI orchestration
│   ├── emotion_analyzer.py  # Emotion detection
│   ├── safety_filter.py     # Content safety
│   └── prompts.py     # System prompts and personality
├── bot/               # Telegram bot interface
│   └── telegram_bot.py
├── models/            # Data models
│   ├── conversation.py
│   └── user.py
├── utils/             # Utilities
│   ├── logger.py      # Structured logging
│   ├── error_handler.py  # Error handling
│   └── rate_limiter.py   # Rate limiting
└── config.py          # Configuration management
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- Azure OpenAI API credentials OR OpenAI API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DenxVil/MitraAI.git
cd MitraAI
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint (if using Azure)
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key (if using Azure)
- `OPENAI_API_KEY`: Your OpenAI API key (if not using Azure)

4. **Run the bot**
```bash
python main.py
```

## 🔧 Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/staging/production) | development |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |
| `MAX_CONVERSATION_HISTORY` | Max messages to keep in context | 10 |
| `RATE_LIMIT_MESSAGES_PER_MINUTE` | Rate limit per user | 20 |
| `ENABLE_CONTENT_MODERATION` | Enable safety filtering | true |

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mitra --cov-report=html

# Run specific test file
pytest tests/unit/test_emotion_analyzer.py -v
```

## 🐳 Docker Deployment

### Build the Docker image
```bash
docker build -t mitra-ai .
```

### Run with Docker
```bash
docker run -d \
  --name mitra-ai \
  --env-file .env \
  mitra-ai
```

### Docker Compose (optional)
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  mitra:
    build: .
    env_file: .env
    restart: unless-stopped
```

Run with: `docker-compose up -d`

## ☁️ Azure Deployment

### Prerequisites
- Azure account
- Azure Container Apps environment
- Azure OpenAI service

### Setup Steps

1. **Create Azure resources**
```bash
# Create resource group
az group create --name mitra-ai-rg --location eastus

# Create Container Apps environment
az containerapp env create \
  --name mitra-env \
  --resource-group mitra-ai-rg \
  --location eastus
```

2. **Configure GitHub Secrets**

Add these secrets to your GitHub repository:
- `AZURE_CREDENTIALS`: Azure service principal credentials
- `AZURE_RESOURCE_GROUP`: Your resource group name
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint

3. **Deploy via GitHub Actions**

Push to the `main` branch to trigger automatic deployment:
```bash
git push origin main
```

The GitHub Actions workflow will:
- Run tests and linting
- Build Docker image
- Push to GitHub Container Registry
- Deploy to Azure Container Apps

## 🤝 Usage

### Telegram Commands

- `/start` - Start conversation with Mitra
- `/help` - Show help and available commands
- `/clear` - Clear conversation history
- `/status` - Show usage statistics

### Example Interactions

**Problem Solving:**
```
You: I'm struggling to decide between two job offers. Can you help me think through this?
Mitra: I'd be happy to help you think through this decision...
```

**Emotional Support:**
```
You: I'm feeling really stressed about my exams.
Mitra: I hear that you're feeling stressed about your exams. That's completely understandable...
```

**Learning:**
```
You: Can you explain how machine learning works?
Mitra: I'd be glad to explain machine learning! Let me break it down step by step...
```

## 📊 Monitoring and Observability

### Structured Logging
All operations are logged with structured data including:
- Correlation IDs for request tracking
- User IDs (anonymized)
- Performance metrics
- Error details with stack traces

### Log Analysis
Logs are output in JSON format (production) or pretty-printed (development) for easy analysis.

### Health Checks
The Docker container includes health checks for monitoring.

## 🔒 Security

### Safety Features
- **Content Moderation**: Filters harmful content
- **Crisis Detection**: Identifies crisis situations and provides resources
- **Rate Limiting**: Prevents abuse
- **Data Privacy**: Minimal data collection, no storage of sensitive information

### Best Practices
- Secrets managed via environment variables
- No credentials in code or logs
- Regular security scanning via Trivy
- Non-root Docker user

## 🛠️ Development

### Code Quality
```bash
# Format code
black mitra/ main.py

# Lint
flake8 mitra/ main.py --max-line-length=100

# Type check
mypy mitra/ main.py
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/my-feature`
2. Implement changes with tests
3. Run tests: `pytest`
4. Format and lint: `black . && flake8`
5. Create pull request

## 📝 Project Structure

```
MitraAI/
├── .github/
│   └── workflows/         # CI/CD workflows
├── mitra/                 # Main application package
│   ├── core/             # Core intelligence engine
│   ├── bot/              # Telegram bot interface
│   ├── models/           # Data models
│   ├── utils/            # Utilities
│   └── config.py         # Configuration
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── pyproject.toml       # Project metadata
└── README.md            # This file
```

## 🤔 Design Decisions

### Why This Architecture?
1. **Modular Design**: Separates concerns for easier testing and maintenance
2. **Emotion-First**: Analyzes emotions before generating responses for empathetic interactions
3. **Safety-First**: Multiple layers of safety checks and crisis detection
4. **Async by Default**: Uses async/await for better performance
5. **Cloud-Native**: Designed for containerized deployment on Azure

### Technology Choices
- **Python 3.11+**: Modern Python with excellent AI/ML ecosystem
- **OpenAI/Azure OpenAI**: State-of-the-art language models
- **python-telegram-bot**: Robust, well-maintained Telegram library
- **structlog**: Structured logging for better observability
- **Pydantic**: Data validation and settings management
- **Docker**: Consistent deployment across environments

## 🚧 Roadmap

### Phase 1 (Current)
- [x] Core AI engine with emotion detection
- [x] Telegram bot interface
- [x] Safety and moderation
- [x] Basic deployment pipeline

### Phase 2 (Planned)
- [ ] Persistent storage (PostgreSQL/MongoDB)
- [ ] Advanced conversation memory
- [ ] Multi-language support
- [ ] Voice message support
- [ ] Web dashboard

### Phase 3 (Future)
- [ ] Fine-tuned models for emotion detection
- [ ] Advanced analytics and insights
- [ ] User personalization
- [ ] Integration with other platforms

## 🐛 Troubleshooting

### Common Issues

**Bot not responding:**
- Check that `TELEGRAM_BOT_TOKEN` is correct
- Verify API credentials are set
- Check logs for errors: `docker logs mitra-ai`

**Rate limit errors:**
- Adjust `RATE_LIMIT_MESSAGES_PER_MINUTE` in `.env`
- Check if user is being rate limited in logs

**AI generation fails:**
- Verify Azure OpenAI/OpenAI credentials
- Check API quota and limits
- Review error logs for specific issues

## 📄 License

This project is available for educational and personal use.

## 🙏 Acknowledgments

- Built with [OpenAI](https://openai.com) and [Azure OpenAI](https://azure.microsoft.com/products/ai-services/openai-service)
- Telegram integration via [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Inspired by the vision of emotionally intelligent AI assistants

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review logs for error details

---

**Note**: Mitra is an AI assistant and should not replace professional mental health support, medical advice, or other professional services. For emergencies, always contact appropriate professionals.
