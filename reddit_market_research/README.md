# Reddit Pain Point Research AI 🔍

An intelligent market research tool that automatically discovers user pain points and generates solutions by analyzing Reddit discussions. This AI-powered system uses advanced workflow orchestration to transform user ideas into comprehensive research reports.

## 🎯 What This Project Does

This application helps entrepreneurs, product managers, and researchers understand what problems people are actually facing by:

1. **Analyzing Reddit Discussions**: Automatically searches through relevant subreddits to find real user conversations
2. **Identifying Pain Points**: Uses AI to extract genuine problems and frustrations from posts and comments
3. **Generating Solutions**: Creates actionable solutions based on both AI analysis and real user suggestions
4. **Creating Reports**: Produces beautiful PDF reports with visualizations and insights

## 🚀 Key Features

- **Streamlit Web Interface**: Easy-to-use web application with real-time progress tracking
- **AI-Powered Analysis**: Uses large language models to understand context and extract meaningful insights
- **Vector Database Storage**: Efficient storage and retrieval of pain points using Qdrant
- **Workflow Orchestration**: Automated 11-step research process using LangGraph
- **PDF Report Generation**: Professional reports with charts and visualizations
- **Duplicate Detection**: Smart deduplication using vector similarity
- **Configurable Parameters**: Customizable search parameters and filtering options

## 🏗️ Architecture Overview

### Core Components

1. **Streamlit Frontend** (`app.py`): Interactive web interface
2. **Reddit Research Agent** (`reddit_agent.py`): Main orchestration engine using LangGraph
3. **LLM Manager** (`llm_manager.py`): Handles all AI model interactions via VLLM
4. **Reddit Manager** (`reddit_manager.py`): Manages Reddit API interactions with rate limiting
5. **Vector Database Manager** (`vector_manager.py`): Handles Qdrant vector storage and retrieval
6. **Report Generator** (`report_manager.py`): Creates visualizations and PDF reports

### Technology Stack

- **VLLM**: High-performance inference server for running large language models locally
- **Qdrant**: Vector database for storing and searching pain point embeddings
- **LangGraph**: Workflow orchestration framework for managing the research pipeline
- **Streamlit**: Web application framework for the user interface
- **Reddit API (PRAW)**: For accessing Reddit data
- **WeasyPrint**: PDF generation from HTML/CSS

## 📋 The 11-Step Research Workflow

The system follows an intelligent workflow orchestrated by LangGraph:

1. **🔤 Generate Keywords**: AI analyzes your project idea to create targeted search terms
2. **📱 Find Subreddits**: Discovers relevant subreddits based on generated keywords
3. **🔍 Search Posts**: Searches Reddit posts across multiple subreddits and strategies
4. **🤖 Filter with AI**: Uses LLM to score and filter posts for relevance (1-10 scale)
5. **💬 Extract Comments**: Retrieves top comments from filtered posts
6. **📊 Analyze Content**: Extracts pain points from posts and comments using AI
7. **🗄️ Store Vectors**: Creates embeddings and stores in Qdrant with duplicate detection
8. **📝 Summarize Pain Points**: Groups and summarizes pain points by themes
9. **🎯 Generate Solution Keywords**: Creates keywords for finding solutions
10. **💡 Generate Solutions**: Finds both AI-generated and Reddit-sourced solutions
11. **📄 Generate Report**: Creates comprehensive PDF report with visualizations

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- VLLM server running (for LLM inference)
- Qdrant database running
- Reddit API credentials

### 1. Clone the Repository

```bash
git clone <repository-url>
cd reddit_market_research
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file with your configuration:

```env
# Reddit API Credentials
REDDIT_CLIENT_ID="your_client_id"
REDDIT_CLIENT_SECRET="your_client_secret"
REDDIT_USERNAME="your_username"
REDDIT_USER_AGENT="your_app_name:v1.0 (by /u/your_username)"

# LLM Configuration (VLLM Server)
MODEL_NAME="your_model_name"
ENDPOINT="http://your_vllm_server:port/v1"

# Qdrant Configuration
QDRANT_HOST="localhost"
QDRANT_PORT=6333

# Embedding Model Configuration
EMBEDDING_MODEL="jinaai/jina-embeddings-v3"
EMBEDDING_VLLM_SERVER_URL="http://your_embedding_server:port/v1"
```

### 4. Set Up External Services

#### VLLM Server Setup
```bash
# Install VLLM
pip install vllm

# Start VLLM server with your model
vllm serve your_model_name --host 0.0.0.0 --port 7001
```

#### Qdrant Database Setup
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
pip install qdrant-client
```

#### Jina Embeddings with VLLM Setup
**Using Docker Run:**
```bash
docker run -d \
  --name jina-embedding \
  --restart unless-stopped \
  -p 7002:8000 \
  --ipc host \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --health-cmd "curl -f http://localhost:8000/v1/models || exit 1" \
  --health-interval 20s \
  --health-timeout 10s \
  --health-retries 3 \
  --health-start-period 90s \
  vllm/vllm-openai:latest \
  --model jinaai/jina-embeddings-v3 \
  --trust-remote-code \
  --dtype=auto \
  --use-tqdm-on-load \
  --tensor-parallel-size=1 \
  --gpu-memory-utilization=0.05
```

**Note**: Update your `.env` file to point to the Jina embedding server:
```env
EMBEDDING_VLLM_SERVER_URL="http://localhost:7002/v1"
```

### 5. Run the Application

```bash
streamlit run app.py
```

## 🎮 How to Use

1. **Open the Application**: Navigate to `http://localhost:8501`
2. **Configure Parameters**: Use the sidebar to adjust search parameters:
   - Number of keywords to generate (2-8)
   - Posts per subreddit (2-10)
   - Minimum post/comment scores
   - Comments to extract per post
3. **Enter Project Idea**: Describe your project in the text area
4. **Start Research**: Click "Start Research" to begin the automated workflow
5. **Monitor Progress**: Watch real-time progress through the 11 steps
6. **Review Results**: Download and view the generated PDF report

## 📊 Configuration Options

### Search Parameters
- **Keywords**: Number of AI-generated search terms (2-8)
- **Posts per Subreddit**: How many posts to analyze per subreddit
- **Minimum Scores**: Filter thresholds for post and comment quality
- **Comments per Post**: Number of top comments to analyze

### AI Model Settings
- **Temperature**: Controls randomness in AI responses
- **Max Tokens**: Maximum length of AI responses
- **Thinking Mode**: Enable/disable AI reasoning steps

## 📁 Project Structure

```
reddit_market_research/
├── app.py                 # Streamlit web interface
├── reddit_agent.py        # Main research orchestration (LangGraph)
├── llm_manager.py         # AI model management (VLLM integration)
├── reddit_manager.py      # Reddit API interactions
├── vector_manager.py      # Qdrant vector database operations
├── report_manager.py      # PDF report generation
├── json_schemas.py        # Pydantic data models
├── prompts.py            # AI prompts and templates
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── LICENSE               # MIT License file
└── projects/             # Generated reports and data
    └── {project_id}/
        ├── final_state.json
        ├── report.pdf
        └── visualizations/
```

## 🔧 Technical Details

### AI Model Integration (VLLM)
- Uses OpenAI-compatible API for seamless integration
- Supports guided JSON generation for structured outputs
- Configurable temperature and token limits
- Thinking mode for complex reasoning tasks

### Vector Database (Qdrant)
- Stores pain point embeddings for similarity search
- Automatic duplicate detection using cosine similarity
- Efficient retrieval with filtering by project ID
- Scalable storage for multiple projects

### Workflow Orchestration (LangGraph)
- State-based workflow management
- Parallel processing where possible
- Error handling and recovery
- Progress tracking and callbacks

## 🚨 Important Notes

- **Rate Limiting**: Reddit API calls are automatically rate-limited
- **Data Privacy**: All processing happens locally on your infrastructure
- **Resource Usage**: LLM inference requires significant computational resources
- **API Limits**: Respect Reddit's API terms of service

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


