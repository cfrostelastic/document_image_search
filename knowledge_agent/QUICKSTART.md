# Quick Start Guide

Get started with the Document Agent in 5 minutes!

## 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Configure Environment

Copy the example configuration:

```bash
cp .env.example .env
```

Edit `.env` with your settings. Choose one of the three providers:

**Option A: Anthropic (Claude)**
```env
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
```

**Option B: OpenAI (GPT-4)**
```env
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4-turbo-preview
```

**Option C: Google Gemini**
```env
AI_PROVIDER=gemini
# Use API key:
GOOGLE_API_KEY=your-google-api-key-here
GEMINI_MODEL=gemini-2.5-flash

# OR use Vertex AI with a service account (recommended for production):
# GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/your/service-account.json
# VERTEX_AI_LOCATION=us-central1
```

Then configure the MCP server paths:

```env
ELASTIC_SEARCH_MCP_COMMAND=python
ELASTIC_SEARCH_MCP_ARGS=/full/path/to/knowledge_agent/document_search_tool/server.py

IMAGE_ANALYSIS_MCP_COMMAND=python
IMAGE_ANALYSIS_MCP_ARGS=/full/path/to/knowledge_agent/image_mcp_service/server.py
```

**Important:** Replace `/full/path/to/knowledge_agent` with your actual path!

## 4. Set Up Elasticsearch (for Example Server)

The example MCP server needs Elasticsearch running:

```bash
# Option 1: Using Docker
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0

# Option 2: Install locally
# Follow https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html
```

Set the Elasticsearch URL:

```bash
export ELASTICSEARCH_URL=http://localhost:9200
```

## 5. Add Test Data

Create a test document in Elasticsearch:

```bash
curl -X POST "http://localhost:9200/documents/_doc/1" -H 'Content-Type: application/json' -d'
{
  "title": "Test Document",
  "content": "This is a test document with important information about Q4 results.",
  "image_path": "/path/to/test/image.png",
  "category": "reports"
}
'
```

## 6. Test Your Setup

Run the diagnostic:

```bash
python test_mcp_tools.py
```

You should see:
```
✓ Connected successfully!
✓ Found X tool(s)
✓ All compatibility checks passed!
```

## 7. Launch the Web UI

```bash
python ui.py
```

Open your browser to: `http://localhost:7860`

## 8. Ask Your First Question

In the web UI, try these example questions:

1. "Search for documents about Q4 results"
2. "What documents do you have?"
3. "Find test documents and tell me what they contain"

Watch the **Tool Activity** panel to see:
- Which tools are called
- What arguments are passed
- What results are returned

## Alternative: Command Line

If you prefer the command line:

```bash
python agent.py
```

Then interact directly:
```
You: What documents do you have?
Agent: [Agent searches and responds...]
```

## Troubleshooting Quick Fixes

### Issue: "ANTHROPIC_API_KEY not found"

```bash
# Check if .env exists
ls -la .env

# Verify it's loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('ANTHROPIC_API_KEY'))"
```

### Issue: "Cannot connect to MCP server"

```bash
# Check paths in .env are absolute (not relative)
# Make sure files exist
ls -l /path/from/ELASTIC_SEARCH_MCP_ARGS
ls -l /path/from/IMAGE_ANALYSIS_MCP_ARGS
```

### Issue: "Connection refused to Elasticsearch"

```bash
# Check if Elasticsearch is running
curl http://localhost:9200

# If not, start it (Docker example)
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.11.0
```

### Issue: "No documents found"

```bash
# Check if documents exist
curl "http://localhost:9200/documents/_search?q=*"

# Add a test document (see step 5 above)
```

## Next Steps

### Using Your Own Data

1. **Index your documents in Elasticsearch:**
   ```bash
   curl -X POST "http://localhost:9200/your_index/_doc" -H 'Content-Type: application/json' -d'
   {
     "title": "Your Document",
     "content": "Your content here...",
     "image_path": "/absolute/path/to/image.png"
   }
   '
   ```

2. **Update the index in your MCP server:**
   ```bash
   export ELASTICSEARCH_INDEX=your_index
   ```

3. **Ask questions about your documents!**

### Customize the Agent

Edit `.env` to customize behavior:

```env
# Change the system prompt
SYSTEM_PROMPT=You are a technical documentation expert. Focus on code examples and diagrams.

# Adjust temperature for more/less creativity
TEMPERATURE=0.3

# Use different models
ANTHROPIC_MODEL=claude-opus-4-5-20251101
```

See [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md) for more examples.

### Create Your Own MCP Servers

The example servers are just templates. You can:

1. Modify `examples/elastic_search_mcp_example.py` for your Elasticsearch setup
2. Modify `examples/image_analysis_mcp_example.py` for custom image processing
3. Create entirely new MCP servers for other data sources

See [MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md) for details.

## Minimal Setup (No Elasticsearch)

If you don't have Elasticsearch, you can create a simple mock MCP server:

Create `simple_search.py`:

```python
#!/usr/bin/env python3
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("simple-search")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_documents",
            description="Search for documents",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "search_documents":
        return [TextContent(
            type="text",
            text=f"Mock result for query: {arguments['query']}\n\nDocument 1:\nTitle: Sample Document\nContent: This is a sample document."
        )]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

Then in `.env`:
```env
ELASTIC_SEARCH_MCP_COMMAND=python
ELASTIC_SEARCH_MCP_ARGS=/path/to/simple_search.py
```

## Learning More

- **[README.md](README.md)** - Full documentation
- **[UI_GUIDE.md](UI_GUIDE.md)** - Detailed UI instructions
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[MCP_SETUP_GUIDE.md](MCP_SETUP_GUIDE.md)** - MCP server configuration
- **[CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)** - Advanced configurations

## Support

If you encounter issues:

1. Run `python test_mcp_tools.py` to diagnose
2. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Verify your `.env` configuration
4. Review the example servers in `examples/`

Happy querying! 🚀
