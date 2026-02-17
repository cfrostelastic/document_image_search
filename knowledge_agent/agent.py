#!/usr/bin/env python3
"""
AI Agent for answering questions about documents using Elastic search and image analysis.
"""

import os
import asyncio
import json
import sys
from pathlib import Path
from anthropic import Anthropic
from openai import OpenAI
from google import genai
from google.genai import types
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config

# Load environment variables (checks local and parent directories)
load_config()


class DocumentAgent:
    """Agent that answers questions about documents using MCP tools."""

    def __init__(self):
        """Initialize the agent with API keys and configuration."""
        self.provider = os.getenv("AI_PROVIDER", "anthropic").lower()

        if self.provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            self.client = Anthropic(api_key=self.api_key)
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=self.api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        elif self.provider == "gemini":
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

            # Check if using Vertex AI (service account) or regular Gemini API
            self.service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

            if self.service_account_file:
                # Use Vertex AI with service account
                if not os.path.exists(self.service_account_file):
                    raise ValueError(f"Service account file not found: {self.service_account_file}")

                # Load service account credentials
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_file,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )

                # Get project ID from service account file
                with open(self.service_account_file, 'r') as f:
                    sa_data = json.load(f)
                    project_id = sa_data.get('project_id')

                # Initialize Vertex AI
                location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
                vertexai.init(
                    project=project_id,
                    location=location,
                    credentials=credentials
                )

                self.use_vertex_ai = True
                self.client = None  # Will create model instance with tools in _answer_with_gemini
            else:
                # Use regular Gemini API with API key
                self.api_key = os.getenv("GOOGLE_API_KEY")
                if not self.api_key:
                    raise ValueError("GOOGLE_API_KEY or GOOGLE_SERVICE_ACCOUNT_FILE must be provided")
                self.use_vertex_ai = False
                self.client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}. Choose 'anthropic', 'openai', or 'gemini'")

        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))

        # System prompt configuration
        self.system_prompt = os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful AI assistant that answers questions about documents. "
            "Follow these steps: 1) Use the search_documents tool to find relevant documents. "
            "2) If search results include an 'Image Path' field, use the analyze_image tool with the image_path parameter to examine the image. "
            "3) If search results include an 'Image URL' field, use the analyze_image_url tool with the image_url parameter. "
            "4) Synthesize information from both document content and image analysis to provide a comprehensive answer."
        )

        # MCP server configurations
        self.elastic_search_config = StdioServerParameters(
            command=os.getenv("ELASTIC_SEARCH_MCP_COMMAND", "node"),
            args=[os.getenv("ELASTIC_SEARCH_MCP_ARGS", "")],
            env=None
        )

        self.image_analysis_config = StdioServerParameters(
            command=os.getenv("IMAGE_ANALYSIS_MCP_COMMAND", "node"),
            args=[os.getenv("IMAGE_ANALYSIS_MCP_ARGS", "")],
            env=None
        )

        self.tools = []
        self.tool_call_callback = None  # Optional callback for UI

    async def connect_mcp_servers(self):
        """Connect to MCP servers and retrieve available tools."""
        print("Connecting to MCP servers...")

        # Connect to Elastic Search MCP server
        async with stdio_client(self.elastic_search_config) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                elastic_tools = await session.list_tools()

                for tool in elastic_tools.tools:
                    self.tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                        "server": "elastic_search"
                    })
                    print(f"  - Loaded tool: {tool.name} (Elastic Search)")

        # Connect to Image Analysis MCP server
        async with stdio_client(self.image_analysis_config) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                image_tools = await session.list_tools()

                for tool in image_tools.tools:
                    self.tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                        "server": "image_analysis"
                    })
                    print(f"  - Loaded tool: {tool.name} (Image Analysis)")

        print(f"Total tools loaded: {len(self.tools)}\n")

    async def call_tool(self, tool_name: str, arguments: dict):
        """Execute a tool call on the appropriate MCP server."""
        # Find which server this tool belongs to
        tool_info = next((t for t in self.tools if t["name"] == tool_name), None)
        if not tool_info:
            raise ValueError(f"Tool {tool_name} not found")

        server = tool_info["server"]

        # Connect to the appropriate server and execute the tool
        if server == "elastic_search":
            async with stdio_client(self.elastic_search_config) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return result
        elif server == "image_analysis":
            async with stdio_client(self.image_analysis_config) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return result
        else:
            raise ValueError(f"Unknown server: {server}")

    async def _answer_with_anthropic(self, question: str) -> str:
        """Answer question using Anthropic's Claude."""
        # Prepare tools for Claude
        claude_tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            }
            for tool in self.tools
        ]

        # Initialize conversation
        messages = [{"role": "user", "content": question}]

        # Agent loop - continue until Claude provides a final answer
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                tools=claude_tools,
                messages=messages
            )

            # Check if we've reached the end
            if response.stop_reason == "end_turn":
                # Extract final text response
                final_response = next(
                    (block.text for block in response.content if hasattr(block, "text")),
                    "I couldn't generate a response."
                )
                return final_response

            # Handle tool use
            if response.stop_reason == "tool_use":
                # Add assistant's response to messages
                messages.append({"role": "assistant", "content": response.content})

                # Process tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        print(f"Using tool: {tool_name}")
                        print(f"  Input: {tool_input}")

                        # Execute tool
                        result = await self.call_tool(tool_name, tool_input)

                        print(f"  Result: {result.content[:200]}...")
                        print()

                        # Call callback if provided (for UI)
                        if self.tool_call_callback:
                            self.tool_call_callback(tool_name, tool_input, str(result.content))

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result.content)
                        })

                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results})
            else:
                # Unexpected stop reason
                return f"Unexpected stop reason: {response.stop_reason}"

    async def _answer_with_openai(self, question: str) -> str:
        """Answer question using OpenAI's GPT."""
        # Prepare tools for OpenAI
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            for tool in self.tools
        ]

        # Initialize conversation with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]

        # Agent loop - continue until we get a final answer
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=openai_tools,
                messages=messages
            )

            message = response.choices[0].message

            # Check if we have tool calls
            if message.tool_calls:
                # Add assistant's response to messages
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })

                # Process tool calls
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)

                    print(f"Using tool: {tool_name}")
                    print(f"  Input: {tool_input}")

                    # Execute tool
                    result = await self.call_tool(tool_name, tool_input)

                    print(f"  Result: {result.content[:200]}...")
                    print()

                    # Call callback if provided (for UI)
                    if self.tool_call_callback:
                        self.tool_call_callback(tool_name, tool_input, str(result.content))

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result.content)
                    })
            else:
                # No tool calls, return the response
                return message.content or "I couldn't generate a response."

    async def _answer_with_gemini(self, question: str) -> str:
        """Answer question using Google's Gemini (via API or Vertex AI)."""
        if self.use_vertex_ai:
            return await self._answer_with_vertex_ai(question)
        else:
            return await self._answer_with_gemini_api(question)

    async def _answer_with_gemini_api(self, question: str) -> str:
        """Answer question using Google's Gemini API."""
        # Prepare tools for Gemini
        gemini_tools = []
        for tool in self.tools:
            # Convert MCP tool schema to Gemini function declaration format
            function_declaration = types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema"]
            )
            gemini_tools.append(function_declaration)

        # Create tool config
        tool_config = types.Tool(function_declarations=gemini_tools)

        # Initialize conversation history
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=question)]
            )
        ]

        # Agent loop - continue until we get a final answer
        while True:
            # Generate response with tools
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    tools=[tool_config],
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )

            # Check if we have function calls
            if response.candidates[0].content.parts:
                has_function_call = any(
                    hasattr(part, 'function_call') and part.function_call
                    for part in response.candidates[0].content.parts
                )

                if has_function_call:
                    # Add assistant's response to conversation
                    contents.append(response.candidates[0].content)

                    # Process all function calls
                    function_response_parts = []

                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            tool_name = function_call.name
                            tool_input = dict(function_call.args)

                            print(f"Using tool: {tool_name}")
                            print(f"  Input: {tool_input}")

                            # Execute tool
                            result = await self.call_tool(tool_name, tool_input)

                            print(f"  Result: {result.content[:200]}...")
                            print()

                            # Call callback if provided (for UI)
                            if self.tool_call_callback:
                                self.tool_call_callback(tool_name, tool_input, str(result.content))

                            # Add function response
                            function_response_parts.append(
                                types.Part(
                                    function_response=types.FunctionResponse(
                                        name=tool_name,
                                        response={"result": str(result.content)}
                                    )
                                )
                            )

                    # Add function responses to conversation
                    contents.append(
                        types.Content(
                            role="user",
                            parts=function_response_parts
                        )
                    )
                else:
                    # No function calls, extract text response
                    text_parts = [
                        part.text for part in response.candidates[0].content.parts
                        if hasattr(part, 'text')
                    ]
                    return ''.join(text_parts) if text_parts else "I couldn't generate a response."
            else:
                # Empty response
                return "I couldn't generate a response."

    async def _answer_with_vertex_ai(self, question: str) -> str:
        """Answer question using Google's Vertex AI."""
        from vertexai.generative_models import (
            FunctionDeclaration,
            Tool,
            Content,
            Part
        )

        # Prepare tools for Vertex AI
        function_declarations = []
        for tool in self.tools:
            # Convert MCP tool schema to Vertex AI function declaration
            function_declaration = FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema"]
            )
            function_declarations.append(function_declaration)

        # Create tool with all function declarations
        vertex_tools = [Tool(function_declarations=function_declarations)]

        # Create model instance with tools
        model = GenerativeModel(
            self.model,
            tools=vertex_tools,
            system_instruction=[self.system_prompt]
        )

        # Start chat session
        chat = model.start_chat()

        # Send initial question
        response = chat.send_message(question)

        # Agent loop - continue until we get a final answer
        while True:
            # Check if we have function calls
            if response.candidates[0].content.parts:
                has_function_call = any(
                    hasattr(part, 'function_call') and part.function_call
                    for part in response.candidates[0].content.parts
                )

                if has_function_call:
                    # Process all function calls
                    function_response_parts = []

                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            tool_name = function_call.name
                            tool_input = dict(function_call.args)

                            print(f"Using tool: {tool_name}")
                            print(f"  Input: {tool_input}")

                            # Execute tool
                            result = await self.call_tool(tool_name, tool_input)

                            print(f"  Result: {result.content[:200]}...")
                            print()

                            # Call callback if provided (for UI)
                            if self.tool_call_callback:
                                self.tool_call_callback(tool_name, tool_input, str(result.content))

                            # Add function response
                            function_response_parts.append(
                                Part.from_function_response(
                                    name=tool_name,
                                    response={"result": str(result.content)}
                                )
                            )

                    # Send function responses back to model
                    response = chat.send_message(function_response_parts)
                else:
                    # No function calls, extract text response
                    text_parts = [
                        part.text for part in response.candidates[0].content.parts
                        if hasattr(part, 'text')
                    ]
                    return ''.join(text_parts) if text_parts else "I couldn't generate a response."
            else:
                # Empty response
                return "I couldn't generate a response."

    async def answer_question(self, question: str) -> str:
        """
        Answer a question about documents using available tools.

        Args:
            question: The user's question about documents

        Returns:
            The agent's answer
        """
        print(f"Question: {question}\n")

        if self.provider == "anthropic":
            return await self._answer_with_anthropic(question)
        elif self.provider == "openai":
            return await self._answer_with_openai(question)
        elif self.provider == "gemini":
            return await self._answer_with_gemini(question)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def run_interactive(self):
        """Run the agent in interactive mode."""
        print("=" * 60)
        print("Document Question Answering Agent")
        print("=" * 60)
        print(f"Provider: {self.provider.upper()}")
        print(f"Model: {self.model}")
        print()

        # Connect to MCP servers
        await self.connect_mcp_servers()

        print("Agent ready! Ask questions about documents (type 'quit' to exit)")
        print("=" * 60)
        print()

        while True:
            try:
                question = input("You: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                answer = await self.answer_question(question)
                print(f"\nAgent: {answer}\n")
                print("-" * 60)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                print("-" * 60)
                print()


async def main():
    """Main entry point."""
    agent = DocumentAgent()
    await agent.run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
