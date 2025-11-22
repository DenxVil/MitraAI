# Mitra AI - Complete Azure Deployment Guide

## Overview

This guide walks you through deploying Mitra AI on Azure using the Azure Portal (Web Interface). Mitra uses a local AI model (Microsoft Phi-3-mini) that runs entirely on Azure infrastructure without requiring external API calls.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Azure Resources Setup via Web Portal](#azure-resources-setup-via-web-portal)
4. [Training Model on Indian Data](#training-model-on-indian-data)
5. [Deployment Steps](#deployment-steps)
6. [Configuration](#configuration)
7. [Monitoring and Scaling](#monitoring-and-scaling)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts
- **Azure Account**: Sign up at [portal.azure.com](https://portal.azure.com)
- **GitHub Account**: For CI/CD integration
- **Telegram Bot Token**: Create via [@BotFather](https://t.me/BotFather)

### Local Requirements (Optional)
- Git
- Docker Desktop (for local testing)
- Python 3.11+ (for local development)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure Cloud                               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Azure Container Apps                                │  │
│  │   ┌──────────────────────────────────────┐          │  │
│  │   │  Mitra AI Container                   │          │  │
│  │   │  - Local Phi-3 Model (3.8B params)   │          │  │
│  │   │  - Emotion Analysis                    │          │  │
│  │   │  - Safety Filter                       │          │  │
│  │   │  - Telegram Bot                        │          │  │
│  │   └──────────────────────────────────────┘          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Azure Container Registry (ACR)                      │  │
│  │   - Stores Docker images                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Azure Key Vault                                     │  │
│  │   - Stores secrets (Telegram token, etc.)            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Azure Monitor / Application Insights                │  │
│  │   - Logs, metrics, and monitoring                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ↑
         │ Telegram API
         │
    ┌────────┐
    │ Users  │
    └────────┘
```

---

## Azure Resources Setup via Web Portal

### Step 1: Create Resource Group

1. **Login to Azure Portal**: Go to [portal.azure.com](https://portal.azure.com)

2. **Navigate to Resource Groups**:
   - Click "Resource groups" in the left menu
   - OR search for "Resource groups" in the top search bar

3. **Create New Resource Group**:
   - Click "+ Create"
   - **Subscription**: Select your subscription
   - **Resource group name**: `mitra-ai-rg`
   - **Region**: `East US` (or your preferred region)
   - Click "Review + Create"
   - Click "Create"

### Step 2: Create Azure Container Registry (ACR)

1. **Navigate to Container Registries**:
   - Search for "Container registries" in the top search bar
   - Click "Container registries"

2. **Create Registry**:
   - Click "+ Create"
   - **Resource group**: Select `mitra-ai-rg`
   - **Registry name**: `mitraairegistry` (must be globally unique)
   - **Location**: `East US` (same as resource group)
   - **SKU**: `Basic` (for development) or `Standard` (for production)
   - Click "Review + Create"
   - Click "Create"

3. **Enable Admin Access** (for GitHub Actions):
   - After creation, go to your registry
   - Click "Access keys" in the left menu
   - Enable "Admin user"
   - Copy **Username** and **Password** (save for later)

### Step 3: Create Azure Key Vault

1. **Navigate to Key Vaults**:
   - Search for "Key vaults" in the top search bar
   - Click "Key vaults"

2. **Create Key Vault**:
   - Click "+ Create"
   - **Resource group**: Select `mitra-ai-rg`
   - **Key vault name**: `mitra-ai-keyvault` (must be globally unique)
   - **Region**: `East US`
   - **Pricing tier**: `Standard`
   - Click "Review + Create"
   - Click "Create"

3. **Add Secrets**:
   - After creation, go to your Key Vault
   - Click "Secrets" in the left menu
   - Click "+ Generate/Import"
   
   **Add Telegram Bot Token**:
   - **Name**: `telegram-bot-token`
   - **Value**: Your Telegram bot token from BotFather
   - Click "Create"

### Step 4: Create Azure Container Apps Environment

1. **Navigate to Container Apps**:
   - Search for "Container apps" in the top search bar
   - Click "Container Apps"

2. **Create Container Apps Environment**:
   - Click "Create"
   - **Resource group**: Select `mitra-ai-rg`
   - **Container app name**: `mitra-ai-app`
   - **Region**: `East US`

3. **Create New Environment**:
   - Click "Create new" for Container Apps Environment
   - **Environment name**: `mitra-ai-environment`
   - **Zone redundancy**: Disabled (for cost savings)
   - Click "Create"

4. **Configure Container**:
   - **Container image source**: Choose "Docker Hub or other registries"
   - **Image type**: Public
   - **Image and tag**: `python:3.11-slim` (temporary, will be updated via CI/CD)
   - **Resource allocation**: 
     - **CPU**: 2.0 cores
     - **Memory**: 4.0 Gi (minimum for AI model)

5. **Environment Variables**:
   - Click "Add" to add environment variables:
     - `ENVIRONMENT`: `production`
     - `LOG_LEVEL`: `INFO`
     - `LOCAL_MODEL_NAME`: `microsoft/Phi-3-mini-4k-instruct`
     - `LOCAL_MODEL_DEVICE`: `cpu`
     - `LOCAL_MODEL_QUANTIZE`: `true`
     - `LOCAL_MODEL_MAX_TOKENS`: `512`

6. **Secrets** (Link to Key Vault):
   - Click "Add" under Secrets
   - **Name**: `telegram-bot-token`
   - **Source**: Azure Key Vault
   - **Key vault secret**: Select your Key Vault and the telegram-bot-token secret

7. **Ingress**:
   - **HTTP Ingress**: Disabled (bot uses webhooks, not HTTP server)
   
8. **Review + Create**:
   - Click "Review + Create"
   - Click "Create"

### Step 5: Set Up GitHub Actions (CI/CD)

1. **Fork the Repository**:
   - Go to the Mitra AI repository on GitHub
   - Click "Fork" to create your own copy

2. **Create Azure Service Principal**:
   - Open Azure Cloud Shell (click >_ icon in top bar)
   - Run this command (replace with your values):
   ```bash
   az ad sp create-for-rbac --name "mitra-ai-sp" \
     --role contributor \
     --scopes /subscriptions/{subscription-id}/resourceGroups/mitra-ai-rg \
     --sdk-auth
   ```
   - Copy the entire JSON output (save for GitHub secrets)

3. **Add GitHub Secrets**:
   - Go to your forked repository on GitHub
   - Click "Settings" → "Secrets and variables" → "Actions"
   - Click "New repository secret" for each:
   
   **Required Secrets**:
   - `AZURE_CREDENTIALS`: Paste the JSON from service principal creation
   - `AZURE_RESOURCE_GROUP`: `mitra-ai-rg`
   - `ACR_USERNAME`: Your ACR username (from Step 2.3)
   - `ACR_PASSWORD`: Your ACR password (from Step 2.3)
   - `ACR_LOGIN_SERVER`: `mitraairegistry.azurecr.io`
   - `CONTAINER_APP_NAME`: `mitra-ai-app`
   - `TELEGRAM_BOT_TOKEN`: Your Telegram bot token

4. **Enable GitHub Actions**:
   - Go to "Actions" tab in your repository
   - Enable workflows if prompted

---

## Training Model on Indian Data

### Overview

The default Phi-3-mini model is multilingual but can be fine-tuned on Indian language data and cultural context for better responses.

### Option 1: Use Pre-trained Multilingual Models

**Recommended Models for Indian Context**:

1. **Ai4Bharat/Airavata** - 7B model trained on Indian languages
   ```python
   # In .env file
   LOCAL_MODEL_NAME=ai4bharat/Airavata
   ```

2. **sarvamai/sarvam-2b-v0.5** - Optimized for Indian languages
   ```python
   LOCAL_MODEL_NAME=sarvamai/sarvam-2b-v0.5
   ```

3. **OpenHathi/OH-2.5_7B_chat** - Hindi-English bilingual
   ```python
   LOCAL_MODEL_NAME=OpenHathi/OH-2.5_7B_chat
   ```

### Option 2: Fine-tune Phi-3 on Indian Data

**Prerequisites**:
- Azure ML Workspace
- GPU compute (Standard_NC6s_v3 or better)
- Training dataset

**Steps**:

1. **Create Azure ML Workspace**:
   - Search for "Machine Learning" in Azure Portal
   - Click "+ Create"
   - **Resource group**: `mitra-ai-rg`
   - **Workspace name**: `mitra-ai-ml-workspace`
   - **Region**: `East US`
   - Click "Review + Create" → "Create"

2. **Prepare Training Data**:
   ```python
   # training_data.jsonl format
   {"messages": [
       {"role": "system", "content": "You are Mitra, an AI assistant."},
       {"role": "user", "content": "भारत की राजधानी क्या है?"},
       {"role": "assistant", "content": "भारत की राजधानी नई दिल्ली है।"}
   ]}
   ```

3. **Launch Azure ML Studio**:
   - Go to your ML Workspace
   - Click "Launch Studio"

4. **Create Compute Instance**:
   - Click "Compute" in left menu
   - Click "+ New" → "Compute Instance"
   - **VM type**: GPU
   - **VM size**: `Standard_NC6s_v3` (6 cores, 112 GB RAM, 1 GPU)
   - Click "Create"

5. **Upload Training Script**:

   Create `train_model.py`:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
   from trl import SFTTrainer
   from datasets import load_dataset
   
   # Load model
   model_name = "microsoft/Phi-3-mini-4k-instruct"
   model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   
   # Load Indian language dataset
   dataset = load_dataset("json", data_files="training_data.jsonl")
   
   # Training arguments
   training_args = TrainingArguments(
       output_dir="./phi3-indian-finetuned",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,
       learning_rate=2e-5,
       fp16=True,
       save_steps=100,
       logging_steps=10,
   )
   
   # Train
   trainer = SFTTrainer(
       model=model,
       args=training_args,
       train_dataset=dataset["train"],
       tokenizer=tokenizer,
       max_seq_length=512,
   )
   
   trainer.train()
   trainer.save_model("./phi3-indian-finetuned-final")
   ```

6. **Run Training**:
   - Upload your training data and script
   - Start a Jupyter notebook or terminal
   - Run: `python train_model.py`
   - Training takes 2-6 hours depending on dataset size

7. **Upload Model to Azure**:
   ```bash
   # Create model storage
   az storage account create --name mitraimodels --resource-group mitra-ai-rg
   
   # Upload model files
   az storage blob upload-batch \
     --account-name mitraimodels \
     --destination models \
     --source ./phi3-indian-finetuned-final
   ```

8. **Update Container App**:
   - Update environment variable:
     ```
     LOCAL_MODEL_NAME=./models/phi3-indian-finetuned-final
     ```

### Recommended Indian Language Datasets

1. **IndicNLP Corpus**: [GitHub](https://github.com/AI4Bharat/indicnlp_corpus)
2. **MASSIVE Dataset**: Multilingual including Hindi
3. **Samanantar**: Parallel corpus for Indian languages
4. **Custom Conversation Data**: Create from customer interactions

---

## Deployment Steps

### Via GitHub Actions (Recommended)

1. **Push Code to GitHub**:
   ```bash
   git add .
   git commit -m "Configure for Azure deployment"
   git push origin main
   ```

2. **GitHub Actions Automatically**:
   - Runs tests
   - Builds Docker image
   - Pushes to Azure Container Registry
   - Deploys to Azure Container Apps

3. **Monitor Deployment**:
   - Go to "Actions" tab in GitHub
   - Click on the running workflow
   - Watch the deployment progress

### Manual Deployment via Azure Portal

1. **Build Docker Image Locally**:
   ```bash
   docker build -t mitra-ai:latest .
   ```

2. **Push to ACR**:
   ```bash
   az acr login --name mitraairegistry
   docker tag mitra-ai:latest mitraairegistry.azurecr.io/mitra-ai:latest
   docker push mitraairegistry.azurecr.io/mitra-ai:latest
   ```

3. **Update Container App**:
   - Go to your Container App in Azure Portal
   - Click "Containers" in left menu
   - Click "Edit and deploy"
   - Update image to: `mitraairegistry.azurecr.io/mitra-ai:latest`
   - Click "Create"

---

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Environment name | `development` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `LOCAL_MODEL_NAME` | HuggingFace model name | `microsoft/Phi-3-mini-4k-instruct` | Yes |
| `LOCAL_MODEL_DEVICE` | Device (cpu/cuda/auto) | `auto` | No |
| `LOCAL_MODEL_QUANTIZE` | Enable 4-bit quantization | `true` | No |
| `LOCAL_MODEL_MAX_TOKENS` | Max tokens to generate | `512` | No |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | - | Yes |
| `RATE_LIMIT_MESSAGES_PER_MINUTE` | Rate limit | `20` | No |
| `MAX_CONVERSATION_HISTORY` | Conversation memory | `10` | No |

### Resource Requirements

**Minimum Requirements**:
- **CPU**: 2 cores
- **Memory**: 4 GB
- **Storage**: 10 GB

**Recommended for Production**:
- **CPU**: 4 cores
- **Memory**: 8 GB
- **Storage**: 20 GB
- **GPU** (optional): Significantly faster inference

### Scaling Configuration

1. **Go to Container App** → "Scale"

2. **Configure Autoscaling**:
   - **Min replicas**: 1
   - **Max replicas**: 5
   - **Scale rule**: CPU Utilization > 70%

3. **Resource Allocation**:
   - Adjust CPU/Memory based on load
   - Monitor with Application Insights

---

## Monitoring and Scaling

### Enable Application Insights

1. **Create Application Insights**:
   - Search for "Application Insights" in Portal
   - Click "+ Create"
   - **Resource group**: `mitra-ai-rg`
   - **Name**: `mitra-ai-insights`
   - Click "Review + Create"

2. **Connect to Container App**:
   - Copy Connection String from Application Insights
   - Add to Container App environment variables:
     ```
     APPLICATIONINSIGHTS_CONNECTION_STRING=<your-connection-string>
     ```

3. **View Metrics**:
   - **Performance**: Response times, throughput
   - **Failures**: Error rates, exceptions
   - **Usage**: User activity, popular commands
   - **Custom**: Emotion detection, safety triggers

### View Logs

1. **Container App Logs**:
   - Go to Container App → "Log stream"
   - Real-time logs from your application

2. **Query Logs in Log Analytics**:
   - Go to Container App → "Logs"
   - Run KQL queries:
   ```kusto
   ContainerAppConsoleLogs_CL
   | where TimeGenerated > ago(1h)
   | where Log_s contains "error"
   | project TimeGenerated, Log_s
   ```

### Performance Optimization

1. **Use GPU for Production**:
   - Significantly faster (5-10x) inference
   - Enable in Container App workload profile

2. **Enable Caching**:
   - Cache frequent responses
   - Reduce model inference calls

3. **Model Quantization**:
   - 4-bit quantization (default): 75% memory reduction
   - Minimal quality impact

---

## Troubleshooting

### Common Issues

#### 1. Model Download Fails

**Symptom**: Container fails to start, logs show model download errors

**Solution**:
```bash
# Pre-download model and include in Docker image
# Add to Dockerfile:
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/Phi-3-mini-4k-instruct')"
```

#### 2. Out of Memory

**Symptom**: Container crashes, OOMKilled in logs

**Solution**:
- Increase memory allocation to 8 GB
- Enable 4-bit quantization
- Reduce `max_new_tokens`

#### 3. Slow Response Times

**Symptom**: Users experience delays

**Solution**:
- Enable GPU (Standard_NC series)
- Reduce conversation history length
- Implement response caching

#### 4. Telegram Webhook Issues

**Symptom**: Bot doesn't respond to messages

**Solution**:
- Check Telegram Bot Token is correct
- Verify Container App is running
- Check logs for errors

### Debug Commands

```bash
# View Container App logs
az containerapp logs show --name mitra-ai-app --resource-group mitra-ai-rg --follow

# Check Container App status
az containerapp show --name mitra-ai-app --resource-group mitra-ai-rg

# Restart Container App
az containerapp update --name mitra-ai-app --resource-group mitra-ai-rg

# View resource usage
az monitor metrics list --resource <resource-id> --metric "CpuPercentage" "MemoryPercentage"
```

### Getting Help

- **Azure Support**: [Azure Portal → Help + Support](https://portal.azure.com/#blade/Microsoft_Azure_Support/HelpAndSupportBlade)
- **GitHub Issues**: [Repository Issues](https://github.com/DenxVil/MitraAI/issues)
- **Documentation**: See README.md and ARCHITECTURE.md

---

## Cost Estimation

### Monthly Costs (East US region)

**Development Environment**:
- Container Apps: $30-50/month
- Container Registry: $5/month
- Key Vault: $1/month
- **Total**: ~$36-56/month

**Production Environment (with GPU)**:
- Container Apps (GPU): $200-400/month
- Container Registry: $20/month
- Key Vault: $1/month
- Application Insights: $10-30/month
- **Total**: ~$231-451/month

**Cost Optimization Tips**:
1. Use CPU-only for development/testing
2. Enable autoscaling to reduce idle costs
3. Use Basic tier for ACR in development
4. Clean up unused resources regularly

---

## Security Best Practices

1. **Use Key Vault** for all secrets
2. **Enable RBAC** on all resources
3. **Restrict network access** using Virtual Networks
4. **Enable diagnostic logging**
5. **Regular security updates** via CI/CD
6. **Monitor for anomalies** in Application Insights

---

## Next Steps

After deployment:

1. **Test the Bot**: Message your bot on Telegram
2. **Monitor Performance**: Check Application Insights
3. **Optimize Model**: Fine-tune on your data
4. **Scale Up**: Add resources as needed
5. **Enable Features**: Add voice, images, etc.

---

## Additional Resources

- **Azure Container Apps**: [Documentation](https://learn.microsoft.com/en-us/azure/container-apps/)
- **Hugging Face Models**: [Model Hub](https://huggingface.co/models)
- **Azure ML**: [Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- **Telegram Bots**: [API Documentation](https://core.telegram.org/bots/api)

---

**Questions?** Open an issue on GitHub or check the main README.md for more information.
