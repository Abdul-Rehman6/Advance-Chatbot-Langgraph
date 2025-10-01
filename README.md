# LangGraph Chatbot with Persistent Memory

A sophisticated AI chatbot powered by LangGraph and OpenAI, featuring persistent conversation history, tool integration, and an elegant Streamlit interface. The chatbot can search the web, perform calculations, fetch stock prices, and maintain context across multiple conversation threads.

## Features

### Core Capabilities
- **Persistent Memory**: All conversations are saved to SQLite with automatic thread management
- **Multi-Tool Integration**: Seamlessly switches between web search, calculations, and stock data
- **Smart Title Generation**: Automatically creates concise, descriptive titles for each conversation
- **Thread Management**: Create, switch between, and resume multiple conversation threads
- **Streaming Responses**: Real-time message streaming for a responsive chat experience
- **Tool Visibility**: Visual indicators show when and which tools are being used

### Available Tools
1. **Tavily Search**: Web search with up to 3 relevant results
2. **Calculator**: Basic arithmetic operations (add, subtract, multiply, divide)
3. **Stock Price Fetcher**: Real-time stock quotes via Alpha Vantage API

## Architecture

### Backend (`langgraph_database_backend.py`)
- Built on LangGraph with OpenAI's GPT-4o-mini model
- SQLite-based checkpointing for conversation persistence
- Custom tools decorated with `@tool` for LangChain integration
- State management using TypedDict and message reducers
- Conditional routing between chat and tool nodes

### Frontend (`streamlit_frontend_database.py`)
- Clean, intuitive Streamlit interface
- Sidebar navigation for conversation threads
- Real-time streaming with tool usage indicators
- Automatic conversation title generation
- Message history reconstruction from checkpoints

### Data Persistence
- **Checkpoints**: LangGraph's SqliteSaver stores full conversation state
- **Thread Summaries**: Custom table for conversation titles and metadata
- **Thread-safe**: Configured for concurrent access across Streamlit sessions

## Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key (for web search)
- Alpha Vantage API key (for stock data)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <project-directory>
```

2. Install dependencies:
```bash
pip install streamlit langchain langchain-core langchain-openai langchain-tavily langgraph python-dotenv requests
```

3. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Note: The Alpha Vantage API key is embedded in the code (free tier). For production use, move it to environment variables.

## Usage

### Starting the Application

Run the Streamlit frontend:
```bash
streamlit run streamlit_frontend_database.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Chatbot

1. **Start a New Chat**: Click "New Chat" in the sidebar to begin a fresh conversation
2. **Ask Questions**: Type your message in the chat input at the bottom
3. **Switch Conversations**: Click any conversation title in the sidebar to resume that thread
4. **Watch Tools Work**: See real-time indicators when the bot uses tools

### Example Queries

```
"What's the current stock price of Apple?"
"Search for the latest developments in AI"
"Calculate 1234 multiplied by 5678"
"What's 150 divided by 6?"
"Tell me about recent tech news"
```

## Project Structure

```
.
├── langgraph_database_backend.py   # LangGraph backend with tools and state management
├── streamlit_frontend_database.py  # Streamlit UI with conversation management
├── chatbot.db                      # SQLite database (auto-generated)
├── .env                            # Environment variables (create this)
└── README.md                       # This file
```

## How It Works

### Message Flow
1. User submits a message through Streamlit
2. Message is added to the current thread's state
3. LangGraph's chat node processes the message with GPT-4o-mini
4. If tools are needed, the graph routes to the tool node
5. Tools execute and return results to the chat node
6. Final response streams back to the user
7. Full conversation state saved to SQLite checkpoint

### Thread Persistence
- Each conversation has a unique `thread_id`
- Checkpoints store the complete message history and state
- Thread summaries table stores auto-generated titles
- On app restart, all threads are loaded from the database

### Title Generation
- Triggered after the first exchange in a new thread
- Uses GPT-4o-mini to create a 3-8 word title
- Follows ChatGPT-style naming conventions
- Sanitizes output to remove quotes, emojis, and excessive punctuation
- Falls back to first user message words if generation fails

## Database Schema

### Checkpoints Table
Created automatically by LangGraph's SqliteSaver:
- Stores serialized conversation state
- Enables time-travel through conversation history
- Allows resuming from any checkpoint

### Thread Summaries Table
```sql
CREATE TABLE thread_summaries (
    thread_id   TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
)
```

## Configuration

### Model Settings
Change the model in `langgraph_database_backend.py`:
```python
llm = ChatOpenAI(model="gpt-4o-mini")  # Change to gpt-4, gpt-3.5-turbo, etc.
```

### Tool Configuration
Modify tool behavior in `langgraph_database_backend.py`:
```python
tavily_search_tool = TavilySearch(max_results=3)  # Adjust search results
```

### Thread Safety
The SQLite connection uses `check_same_thread=False` for Streamlit compatibility. For production deployments, consider PostgreSQL with LangGraph's PostgresSaver.

## Troubleshooting

### Database Locked Error
If you encounter database lock errors:
- Ensure only one instance is running
- Close any SQLite database browsers
- Restart the application

### API Key Errors
Verify your `.env` file contains valid API keys:
```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### Missing Dependencies
Install all required packages:
```bash
pip install -r requirements.txt  # If you create one
```

## Advanced Features

### Custom Tools
Add your own tools to `langgraph_database_backend.py`:

```python
@tool
def your_custom_tool(param: str) -> dict:
    """Tool description for the LLM."""
    # Your implementation
    return {"result": "data"}

# Add to tools list
tools = [tavily_search_tool, calculator, get_stock_price, your_custom_tool]
```

### Multiple Threads
The system automatically manages multiple conversation threads. Each thread maintains its own:
- Message history
- Tool usage context
- Checkpoints for state recovery

## Performance Considerations

- **Streaming**: Uses LangGraph's stream mode for real-time responses
- **Lazy Loading**: Conversations load only when accessed
- **Efficient Storage**: SQLite provides fast local persistence
- **Connection Pooling**: Single database connection shared across requests

## Security Notes

- Never commit `.env` files to version control
- Rotate API keys regularly
- Use environment variables for all secrets
- Consider rate limiting for production deployments
- Implement user authentication for multi-user scenarios

## Future Enhancements

Potential improvements for this chatbot:
- Export conversation history to PDF/TXT
- Conversation search functionality
- Delete individual threads
- Custom tool configurations per thread
- Multi-user support with authentication
- Cloud database integration (PostgreSQL)
- Voice input/output capabilities
- File upload and analysis tools

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is provided as-is for educational and development purposes.

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Orchestration framework
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [Streamlit](https://streamlit.io/) - Web interface
- [OpenAI](https://openai.com/) - Language model
- [Tavily](https://tavily.com/) - Web search API
- [Alpha Vantage](https://www.alphavantage.co/) - Stock market data

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Review the code documentation
- Check LangGraph and Streamlit documentation

---

Made with LangGraph, OpenAI, and Streamlit