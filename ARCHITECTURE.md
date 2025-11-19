# Mitra AI - Architecture Documentation

## Overview

Mitra AI is built with a modular, layered architecture designed for maintainability, scalability, and emotional intelligence. The system combines advanced AI capabilities with robust safety features and comprehensive observability.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface Layer                  │
│                    (Telegram Bot)                        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Bot Interface Layer                     │
│  - Command handlers                                      │
│  - Message routing                                       │
│  - Rate limiting                                         │
│  - User session management                               │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               Core Intelligence Layer                    │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Emotion      │  │   Safety     │  │   Prompt     │ │
│  │  Analyzer     │  │   Filter     │  │   Builder    │ │
│  └───────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │          MitraEngine (Orchestrator)              │  │
│  │  - Conversation management                       │  │
│  │  - AI API integration                            │  │
│  │  - Response generation                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   AI Service Layer                       │
│            (Azure OpenAI / OpenAI)                       │
└─────────────────────────────────────────────────────────┘

         Cross-cutting Concerns (All Layers):
    ┌────────────────────────────────────────────┐
    │  - Structured Logging                      │
    │  - Error Handling                          │
    │  - Configuration Management                │
    │  - Monitoring & Observability              │
    └────────────────────────────────────────────┘
```

## Component Details

### 1. Bot Interface Layer (`mitra/bot/`)

**Purpose**: Handles all interactions with Telegram and manages user sessions.

**Components**:
- `telegram_bot.py`: Main bot implementation
  - Command handlers (/start, /help, /clear, /status)
  - Message routing
  - User management
  - Rate limiting integration

**Key Design Decisions**:
- Async/await for non-blocking operations
- Separate error handler for graceful failure
- Correlation ID tracking for all requests
- Rate limiting at bot level before AI processing

### 2. Core Intelligence Layer (`mitra/core/`)

**Purpose**: Orchestrates AI responses with emotion awareness and safety.

#### 2.1 MitraEngine (`engine.py`)

The central orchestrator that:
- Manages conversation state
- Coordinates between emotion analysis, safety, and AI generation
- Handles retries and error recovery
- Builds prompts with emotional context

**Processing Flow**:
```
Message Received
    ↓
Safety Check → [FAIL] → Return boundary response
    ↓ [PASS]
Crisis Detection → [YES] → Return crisis resources
    ↓ [NO]
Emotion Analysis
    ↓
Add to Conversation
    ↓
Build API Messages (with context)
    ↓
Call AI API (with retries)
    ↓
Generate Response
    ↓
Add Response to Conversation
    ↓
Return Response
```

#### 2.2 EmotionAnalyzer (`emotion_analyzer.py`)

Analyzes emotional content in user messages using:
- Pattern matching for emotion keywords
- Sentiment classification (positive, negative, neutral)
- Intensity calculation based on language markers
- Crisis pattern detection
- Urgency level assessment

**Detected Emotions**:
- Joy, Sadness, Anger, Fear
- Stress, Confusion, Gratitude, Excitement

**Output**: `EmotionalContext` object with:
- Sentiment
- List of emotions
- Intensity (0.0-1.0)
- Support needed flag
- Urgency level

#### 2.3 SafetyFilter (`safety_filter.py`)

Multi-layered safety system:
1. **Content Moderation**: Blocks harmful requests
2. **Crisis Detection**: Identifies urgent situations
3. **Professional Referral**: Recommends experts when needed
4. **Boundary Setting**: Polite refusals for inappropriate content

**Safety Categories**:
- Harmful content
- Crisis situations
- Professional help needed
- Inappropriate requests

#### 2.4 PromptBuilder (`prompts.py`)

Constructs prompts that define Mitra's personality and behavior:

**System Prompt Structure**:
```
Base Personality
    ↓
Core Capabilities
    ↓
Response Guidelines
    ↓
Safety Rules
    ↓
Emotional Context (dynamic)
```

### 3. Data Models (`mitra/models/`)

**Purpose**: Type-safe data structures using Pydantic.

#### Models:
- `Message`: Individual message with role, content, timestamp, emotion
- `Conversation`: Thread of messages with history management
- `User`: User profile and activity tracking
- `EmotionalContext`: Emotion analysis results

**Design Principles**:
- Immutable IDs (UUID)
- Automatic timestamps
- JSON serializable
- OpenAI format conversion

### 4. Utilities (`mitra/utils/`)

#### 4.1 Logger (`logger.py`)

Structured logging with:
- JSON output (production) / Pretty console (development)
- Correlation ID tracking
- Context variables
- Multiple log levels
- Stack trace capture

#### 4.2 ErrorHandler (`error_handler.py`)

Centralized error handling:
- Custom exception types (`MitraError`)
- Error categorization
- User-friendly message generation
- Full context logging
- Original error preservation

#### 4.3 RateLimiter (`rate_limiter.py`)

Simple in-memory rate limiting:
- Per-user request tracking
- Sliding time window
- Configurable limits
- Remaining request queries

**Note**: For production with multiple instances, replace with Redis-based limiter.

### 5. Configuration (`mitra/config.py`)

Environment-based configuration using Pydantic Settings:
- Type validation
- Default values
- Environment variable loading
- Configuration validation
- Computed properties

## Data Flow

### Message Processing Flow

```
1. User sends message via Telegram
   ↓
2. Telegram API → Bot receives update
   ↓
3. Bot extracts user info and message
   ↓
4. Rate limiter checks user quota
   ↓
5. Engine.process_message() called
   ↓
6. Safety filter checks content
   ↓
7. Emotion analyzer processes message
   ↓
8. Message added to conversation with context
   ↓
9. Prompt builder creates API messages
   ↓
10. AI API called with retries
   ↓
11. Response generated and validated
   ↓
12. Response added to conversation
   ↓
13. Response sent back to user via Telegram
```

### Conversation State Management

```
In-Memory Storage (Current):
  conversations: Dict[conversation_id, Conversation]
  users: Dict[user_id, User]

Future (Persistent):
  Database (PostgreSQL/MongoDB)
  Cache (Redis)
```

## AI Integration

### Model Selection
- **Azure OpenAI**: Primary (better control, regional compliance)
- **OpenAI API**: Fallback (simpler setup)

### API Call Strategy
- **Model**: GPT-4 (or Azure deployment)
- **Temperature**: 0.7 (balanced creativity/consistency)
- **Max Tokens**: 800 (concise but complete)
- **Top P**: 0.9
- **Frequency Penalty**: 0.3 (reduce repetition)
- **Presence Penalty**: 0.3 (encourage diversity)

### Retry Strategy
- **Attempts**: 3
- **Backoff**: Exponential (2s, 4s, 8s)
- **Jitter**: Built-in via tenacity

## Safety Architecture

### Multi-Layer Safety

```
Layer 1: Input Validation
  - Length checks
  - Format validation

Layer 2: Content Moderation
  - Harmful pattern detection
  - Inappropriate content filtering

Layer 3: Crisis Detection
  - Self-harm indicators
  - Emergency situations
  - Immediate resource provision

Layer 4: Response Validation
  - Ensure appropriate tone
  - Verify safety compliance
```

### Crisis Response Protocol

```
Crisis Detected
    ↓
1. Log with high priority
    ↓
2. Return immediate resources
    ↓
3. Include professional contacts
    ↓
4. Maintain supportive but clear boundaries
    ↓
5. Continue conversation if user wishes
```

## Observability

### Logging Strategy

**Development**:
- Pretty console output with colors
- Detailed debug information
- Full stack traces

**Production**:
- JSON structured logs
- Log aggregation ready
- Performance metrics
- Error tracking

### Key Metrics (Logged)
- Request latency
- AI API call duration
- Token usage
- Error rates by category
- Rate limit hits
- Emotion distribution
- Crisis detection frequency

### Correlation IDs
Every request gets a unique correlation ID that flows through:
1. Bot handler
2. Engine processing
3. Component calls
4. AI API requests
5. Error logging
6. Response delivery

## Deployment Architecture

### Container Strategy

```
Docker Image
    ↓
GitHub Container Registry
    ↓
Azure Container Apps
    ↓
Auto-scaling based on load
```

### CI/CD Pipeline

```
Code Push
    ↓
GitHub Actions Triggered
    ↓
[Parallel]
├─ Run Tests
├─ Run Linting
└─ Security Scan
    ↓
Build Docker Image
    ↓
Push to Registry
    ↓
Deploy to Azure
    ↓
Health Check
    ↓
Production Traffic
```

### Environment Management
- **Development**: Local + .env
- **Staging**: Azure Container Apps + Key Vault
- **Production**: Azure Container Apps + Key Vault + Monitoring

## Scalability Considerations

### Current Limitations (v0.1.0)
1. In-memory conversation storage
2. In-memory rate limiting
3. Single-instance deployment

### Future Scaling Plan
1. **Database**: PostgreSQL for conversations, users
2. **Cache**: Redis for rate limiting, session state
3. **Queue**: Azure Service Bus for async processing
4. **CDN**: Azure CDN for static content (if web added)
5. **Multiple Instances**: Load balancing across containers

## Security Architecture

### Secrets Management
- Environment variables (dev)
- Azure Key Vault (production)
- No secrets in code/logs
- Regular rotation

### Data Privacy
- Minimal data collection
- No persistent message storage (current)
- User anonymization in logs
- GDPR compliance ready

### Network Security
- HTTPS only
- Webhook verification
- Rate limiting
- DDoS protection via Azure

## Testing Strategy

### Unit Tests
- Core components isolated
- Mock external dependencies
- Fast execution
- High coverage target (>80%)

### Integration Tests
- End-to-end flows
- Real API calls (test endpoints)
- Performance benchmarks

### Manual Testing
- Telegram bot interaction
- Edge cases
- User experience validation

## Future Architecture Enhancements

### Phase 2
1. **Persistent Storage**: Database integration
2. **Advanced Memory**: Long-term user context
3. **Multi-Language**: i18n support
4. **Voice Support**: Speech-to-text integration

### Phase 3
1. **Fine-tuned Models**: Custom emotion detection
2. **Analytics Dashboard**: Usage insights
3. **A/B Testing**: Response optimization
4. **Advanced Personalization**: User-specific models

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| AI | Azure OpenAI / OpenAI | Language model |
| Bot | python-telegram-bot | Telegram integration |
| Logging | structlog | Structured logging |
| Config | pydantic-settings | Configuration management |
| Validation | pydantic | Data validation |
| Testing | pytest | Test framework |
| Async | asyncio | Async operations |
| Retry | tenacity | Retry logic |
| Container | Docker | Containerization |
| Cloud | Azure Container Apps | Hosting |
| CI/CD | GitHub Actions | Automation |
| Monitoring | Azure App Insights | Observability |

## Design Principles

1. **Modularity**: Clear separation of concerns
2. **Type Safety**: Full type hints throughout
3. **Async First**: Non-blocking operations
4. **Error Recovery**: Graceful degradation
5. **Observability**: Comprehensive logging
6. **Safety**: Multiple protection layers
7. **Scalability**: Designed for growth
8. **Maintainability**: Clean, documented code
