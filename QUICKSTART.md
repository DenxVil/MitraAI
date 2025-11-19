# Mitra AI - Quick Start Guide

Get Mitra AI up and running in 5 minutes!

## Prerequisites

- Python 3.11 or higher
- A Telegram account
- OpenAI API key OR Azure OpenAI credentials

## Step 1: Get a Telegram Bot Token

1. Open Telegram and search for [@BotFather](https://t.me/BotFather)
2. Send `/newbot` command
3. Follow the prompts to create your bot
4. Copy the API token (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

## Step 2: Get OpenAI Credentials

### Option A: OpenAI (Easiest)
1. Sign up at [OpenAI](https://platform.openai.com)
2. Go to API keys section
3. Create a new API key
4. Copy the key (starts with `sk-`)

### Option B: Azure OpenAI
1. Create an Azure account
2. Create an Azure OpenAI resource
3. Deploy a GPT-4 model
4. Note your endpoint and API key

## Step 3: Install Mitra

```bash
# Clone the repository
git clone https://github.com/DenxVil/MitraAI.git
cd MitraAI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your favorite editor
nano .env  # or vim, code, etc.
```

### Minimal Configuration (OpenAI)
```env
TELEGRAM_BOT_TOKEN=your_telegram_token_here
OPENAI_API_KEY=your_openai_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Minimal Configuration (Azure OpenAI)
```env
TELEGRAM_BOT_TOKEN=your_telegram_token_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_key_here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## Step 5: Run Mitra

```bash
python main.py
```

You should see:
```
🤖 Mitra AI is now running...
Environment: development
Model: gpt-4

Press Ctrl+C to stop.
```

## Step 6: Test Your Bot

1. Open Telegram
2. Search for your bot by username (the one you set with BotFather)
3. Click "Start" or send `/start`
4. You should get a welcome message from Mitra!
5. Try sending a message like "Hello, how are you?"

## Example Interactions

### Problem Solving
```
You: I need to decide between two job offers. Can you help?
Mitra: I'd be happy to help you think through this important decision...
```

### Emotional Support
```
You: I'm feeling really stressed about my presentation tomorrow.
Mitra: I hear that you're feeling stressed. That's completely normal before a big presentation...
```

### Learning
```
You: Can you explain what machine learning is?
Mitra: I'd be glad to explain! Machine learning is...
```

## Useful Commands

- `/start` - Start or restart conversation
- `/help` - Show help message
- `/clear` - Clear conversation history
- `/status` - Show your usage stats

## Troubleshooting

### Bot not responding?

**Check logs:**
The console will show what's happening. Look for error messages.

**Common issues:**
- Wrong bot token → Double-check your TELEGRAM_BOT_TOKEN
- Wrong API key → Verify your OpenAI/Azure credentials
- Network issues → Check your internet connection

### "Rate limit exceeded"?
You're sending messages too fast. Wait a moment and try again.

### "AI service error"?
- Check your API credits/quota
- Verify your credentials are correct
- Ensure the API endpoint is accessible

## Next Steps

### Customize Mitra

Edit `.env` to adjust:
- `MAX_CONVERSATION_HISTORY` - How many messages to remember (default: 10)
- `RATE_LIMIT_MESSAGES_PER_MINUTE` - Rate limit (default: 20)
- `LOG_LEVEL` - Logging detail (DEBUG, INFO, WARNING, ERROR)

### Deploy to Production

See [README.md](README.md#-azure-deployment) for Azure deployment instructions.

### Explore the Code

- `mitra/core/engine.py` - Main AI orchestration
- `mitra/core/prompts.py` - Mitra's personality and instructions
- `mitra/core/emotion_analyzer.py` - Emotion detection logic
- `mitra/core/safety_filter.py` - Safety and crisis detection

### Contribute

Want to improve Mitra? See [CONTRIBUTING.md](CONTRIBUTING.md)!

## Advanced Configuration

### Enable Content Moderation
```env
ENABLE_CONTENT_MODERATION=true
```

### Change Model Temperature
Edit `mitra/core/engine.py` and adjust the `temperature` parameter in the API call.

### Add Custom System Prompts
Edit `mitra/core/prompts.py` to customize Mitra's personality.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=mitra --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Docker (Optional)

```bash
# Build image
docker build -t mitra-ai .

# Run container
docker run -d --name mitra --env-file .env mitra-ai

# View logs
docker logs -f mitra

# Stop container
docker stop mitra
```

## Getting Help

- **Documentation**: See [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md)
- **Issues**: Check [GitHub Issues](https://github.com/DenxVil/MitraAI/issues)
- **Questions**: Open a GitHub Discussion

## Safety Reminders

⚠️ **Important**:
- Never share your API keys publicly
- Don't commit `.env` file to git
- Mitra is an AI assistant, not a replacement for professional help
- For mental health crises, contact professional services

## Success Checklist

- [ ] Python 3.11+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] `.env` file configured with valid credentials
- [ ] Bot running without errors
- [ ] Can send `/start` and get a response
- [ ] Can have a conversation with Mitra

Congratulations! You now have Mitra AI running! 🎉

---

**Need more help?** Check the full [README.md](README.md) or open an issue.
