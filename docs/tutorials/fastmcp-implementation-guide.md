# FastMCP Implementation Guide

## Overview

FastMCP is the high-level, Pythonic framework for building MCP servers. It provides decorator-based APIs that make server development intuitive and powerful.

## Core Architecture

### Server Creation
```python
from mcp.server.fastmcp import FastMCP

# Basic server
mcp = FastMCP("My Server")

# Server with dependencies
mcp = FastMCP("My Server", dependencies=["pandas", "numpy"])
```

### Server Capabilities

FastMCP servers can implement three core primitives:

| Primitive | Decorator | Purpose | Control |
|-----------|-----------|---------|---------|
| **Tools** | `@mcp.tool()` | Execute actions/functions | Model-controlled |
| **Resources** | `@mcp.resource()` | Provide contextual data | Application-controlled |
| **Prompts** | `@mcp.prompt()` | Interactive templates | User-controlled |

## Tools Implementation

Tools are functions that LLMs can call to perform actions:

```python
@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters"""
    return weight_kg / (height_m**2)

@mcp.tool(title="Weather Fetcher")
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
```

### Structured Output

FastMCP automatically handles structured output for properly typed return values:

```python
from pydantic import BaseModel, Field

class WeatherData(BaseModel):
    temperature: float = Field(description="Temperature in Celsius")
    humidity: float = Field(description="Humidity percentage")
    condition: str
    wind_speed: float

@mcp.tool()
def get_weather(city: str) -> WeatherData:
    """Get structured weather data"""
    return WeatherData(
        temperature=22.5, 
        humidity=65.0, 
        condition="partly cloudy", 
        wind_speed=12.3
    )
```

**Supported Types for Structured Output:**
- Pydantic models (BaseModel subclasses)
- TypedDicts
- Dataclasses with type hints
- `dict[str, T]` where T is JSON-serializable
- Primitive types (wrapped in `{"result": value}`)

## Resources Implementation

Resources expose data to LLMs, similar to GET endpoints:

```python
@mcp.resource("config://app", title="Application Configuration")
def get_config() -> str:
    """Static configuration data"""
    return "App configuration here"

@mcp.resource("users://{user_id}/profile", title="User Profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user data with URI templating"""
    return f"Profile data for user {user_id}"
```

## Prompts Implementation

Prompts provide reusable templates for LLM interactions:

```python
from mcp.server.fastmcp.prompts import base

@mcp.prompt(title="Code Review")
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"

@mcp.prompt(title="Debug Assistant")
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]
```

## Context Support

The Context object provides access to MCP capabilities:

```python
from mcp.server.fastmcp import FastMCP, Context

@mcp.tool()
async def long_task(files: list[str], ctx: Context) -> str:
    """Process multiple files with progress tracking"""
    for i, file in enumerate(files):
        ctx.info(f"Processing {file}")
        await ctx.report_progress(i, len(files))
        data, mime_type = await ctx.read_resource(f"file://{file}")
    return "Processing complete"
```

## Lifecycle Management

FastMCP supports application lifecycle management:

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

@dataclass
class AppContext:
    db: Database

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context"""
    # Initialize on startup
    db = await Database.connect()
    try:
        yield AppContext(db=db)
    finally:
        # Cleanup on shutdown
        await db.disconnect()

# Pass lifespan to server
mcp = FastMCP("My App", lifespan=app_lifespan)

@mcp.tool()
def query_db() -> str:
    """Tool that uses initialized resources"""
    ctx = mcp.get_context()
    db = ctx.request_context.lifespan_context["db"]
    return db.query()
```

## Development and Testing

### Development Mode
```bash
# Basic development server with inspector
mcp dev server.py

# With additional dependencies
mcp dev server.py --with pandas --with numpy

# Mount local code for development
mcp dev server.py --with-editable .
```

### Claude Desktop Integration
```bash
# Install in Claude Desktop
mcp install server.py

# Custom name and environment variables
mcp install server.py --name "My Analytics Server"
mcp install server.py -v API_KEY=abc123 -v DB_URL=postgres://...
mcp install server.py -f .env
```

### Direct Execution
```python
if __name__ == "__main__":
    mcp.run()
```

## Transport Options

### Standard Transport (stdio)
Default transport for local servers:
```python
mcp.run()  # Uses stdio by default
```

### HTTP Transport
For remote servers:
```python
# Stateless HTTP server
mcp = FastMCP("My Server", stateless_http=True)
mcp.run(transport="streamable-http")
```

## Authentication

FastMCP supports OAuth 2.1 authentication:

```python
from mcp.server.auth.provider import TokenVerifier, TokenInfo
from mcp.server.auth.settings import AuthSettings

class MyTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> TokenInfo:
        # Implement token validation logic
        ...

mcp = FastMCP(
    "My App",
    token_verifier=MyTokenVerifier(),
    auth=AuthSettings(
        issuer_url="https://auth.example.com",
        resource_server_url="http://localhost:3001",
        required_scopes=["mcp:read", "mcp:write"],
    ),
)
```

## Best Practices

1. **Type Hints**: Always use proper type hints for structured output
2. **Error Handling**: Implement comprehensive error handling in tools
3. **Resource Caching**: Cache expensive resource computations
4. **Async Operations**: Use async/await for I/O-bound operations
5. **Security**: Validate inputs and implement proper authentication
6. **Documentation**: Provide clear docstrings for all tools and resources

## Example: Complete Server

```python
import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("Weather Service")

class WeatherData(BaseModel):
    temperature: float
    condition: str
    city: str

@mcp.resource("weather://current")
def current_conditions() -> str:
    """Current weather conditions resource"""
    return "Current weather data available"

@mcp.tool()
async def get_weather(city: str) -> WeatherData:
    """Fetch current weather for a city"""
    # Simulate API call
    return WeatherData(
        temperature=22.5,
        condition="sunny",
        city=city
    )

@mcp.prompt()
def weather_analysis(city: str) -> str:
    """Generate weather analysis prompt"""
    return f"Analyze the weather patterns for {city} and provide insights."

if __name__ == "__main__":
    mcp.run()
```

This guide provides the foundation for building robust MCP servers with FastMCP. The framework's decorator-based approach makes it easy to expose functionality while maintaining type safety and proper MCP protocol compliance.