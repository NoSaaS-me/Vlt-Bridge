# Tool Call Parsers

Clean, maintainable architecture for parsing tool calls from different LLM models using the **Strategy Pattern**.

## Overview

Some LLM models don't properly support OpenAI-style function calling and instead output XML-style tool invocations in their text responses. This package provides a flexible, extensible system for parsing these various formats and converting them to the standard OpenAI function calling format.

## Architecture

```
tool_parsers/
├── base.py          # Abstract base classes and data structures
├── standard.py      # Standard <function_calls> format
├── anthropic.py     # Anthropic standalone <invoke> format
├── deepseek.py      # DeepSeek <｜DSML｜...> format
├── generic.py       # Fallback parser with flexible matching
├── chain.py         # Parser chain orchestration
├── __init__.py      # Package exports
└── README.md        # This file
```

## Components

### Base Classes (`base.py`)

#### `ParsedToolCall`
Intermediate representation of a parsed tool call.

```python
@dataclass
class ParsedToolCall:
    name: str                    # Tool/function name
    arguments: Dict[str, Any]    # Parameter dict
    raw_xml: str                 # Original XML for debugging

    def to_openai_format(self, call_id: str) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
```

#### `ToolCallParser`
Abstract base class for all parsers.

```python
class ToolCallParser(ABC):
    @abstractmethod
    def can_parse(self, content: str) -> bool:
        """Check if this parser can handle the content."""

    @abstractmethod
    def parse(self, content: str) -> Tuple[List[ParsedToolCall], str]:
        """Extract tool calls and return cleaned content."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Parser name for logging."""
```

### Concrete Parsers

#### StandardXMLParser (`standard.py`)
Handles the most common format:

```xml
<function_calls>
<invoke name="tool_name">
<parameter name="param_name">value</parameter>
</invoke>
</function_calls>
```

#### AnthropicXMLParser (`anthropic.py`)
Handles standalone invoke elements:

```xml
<invoke name="tool_name">
<parameter name="param_name">value</parameter>
</invoke>
```

#### DeepSeekXMLParser (`deepseek.py`)
Handles DeepSeek's special format with `｜DSML｜` markers:

```xml
<｜DSML｜function_calls>
<｜DSML｜invoke name="tool_name">
<｜DSML｜parameter name="param_name">value</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>
```

#### GenericXMLParser (`generic.py`)
Fallback parser with flexible regex patterns for handling variations:
- Tags with prefixes: `<prefix:function_calls>`
- Tags with special characters
- Malformed spacing

### Parser Chain (`chain.py`)

Implements Chain of Responsibility pattern, trying parsers in priority order:

```python
class ToolCallParserChain:
    def __init__(self):
        self.parsers = [
            DeepSeekXMLParser(),    # Most specific - check first
            StandardXMLParser(),     # Common format
            AnthropicXMLParser(),    # Standalone invokes
            GenericXMLParser(),      # Fallback - most permissive
        ]

    def parse(self, content: str) -> Tuple[List[Dict[str, Any]], str]:
        """Try each parser until one succeeds."""
```

## Usage

### Basic Usage

```python
from src.services.tool_parsers import ToolCallParserChain

# Create parser chain
chain = ToolCallParserChain()

# Parse content from LLM
model_output = """
<function_calls>
<invoke name="search_code">
<parameter name="query">authentication</parameter>
<parameter name="limit">5</parameter>
</invoke>
</function_calls>
"""

# Extract tool calls
tool_calls, cleaned_content = chain.parse(model_output)

# tool_calls is in OpenAI format:
# [
#     {
#         "id": "xml_call_abc123",
#         "type": "function",
#         "function": {
#             "name": "search_code",
#             "arguments": '{"query": "authentication", "limit": 5}'
#         }
#     }
# ]
```

### Integration with Oracle Agent

The `oracle_agent.py` uses the parser chain as a fallback when models output XML instead of proper function calls:

```python
from .tool_parsers import ToolCallParserChain

def _parse_xml_tool_calls(content: str) -> Tuple[List[Dict[str, Any]], str]:
    """Parse XML-style function calls from content."""
    parser_chain = ToolCallParserChain()
    return parser_chain.parse(content)
```

### Using Individual Parsers

```python
from src.services.tool_parsers.standard import StandardXMLParser

parser = StandardXMLParser()

# Check if content matches this parser
if parser.can_parse(content):
    # Parse using this specific parser
    calls, cleaned = parser.parse(content)
```

## Parameter Type Inference

All parsers automatically infer parameter types:

- **Booleans**: `"true"` → `True`, `"false"` → `False`
- **Integers**: `"123"` → `123`
- **JSON**: `'{"key": "value"}'` → `{"key": "value"}`
- **Strings**: Everything else remains as string

Example:
```xml
<parameter name="enabled">true</parameter>           <!-- bool -->
<parameter name="port">8080</parameter>               <!-- int -->
<parameter name="config">{"debug": true}</parameter>  <!-- dict -->
<parameter name="name">test</parameter>               <!-- str -->
```

## Content Cleaning

The parser removes all XML markup and normalizes whitespace:

**Input:**
```
I'll help you.

<function_calls>
<invoke name="search">
<parameter name="q">test</parameter>
</invoke>
</function_calls>


Here's what I found.
```

**Output:**
```
I'll help you.

Here's what I found.
```

## Testing

Comprehensive unit tests in `tests/unit/test_tool_parsers.py`:

```bash
# Run all parser tests
pytest tests/unit/test_tool_parsers.py -v

# Run specific test class
pytest tests/unit/test_tool_parsers.py::TestStandardXMLParser -v

# Run with coverage
pytest tests/unit/test_tool_parsers.py --cov=src.services.tool_parsers
```

Test coverage includes:
- ✅ Each parser format
- ✅ Parameter type inference
- ✅ Multiple tool calls in one block
- ✅ Edge cases (empty parameters, special characters)
- ✅ Chain fallback behavior
- ✅ OpenAI format conversion
- ✅ Content cleaning

## Adding New Parsers

To add support for a new model's format:

1. **Create a new parser class** in `tool_parsers/`:

```python
# mymodel.py
from .base import ToolCallParser, ParsedToolCall

class MyModelParser(ToolCallParser):
    @property
    def name(self) -> str:
        return "MyModelParser"

    def can_parse(self, content: str) -> bool:
        # Check if content matches your format
        return "<mymodel>" in content

    def parse(self, content: str) -> Tuple[List[ParsedToolCall], str]:
        # Extract tool calls
        # Return (calls, cleaned_content)
        pass
```

2. **Add to chain** in `chain.py`:

```python
from .mymodel import MyModelParser

class ToolCallParserChain:
    def __init__(self):
        self.parsers = [
            MyModelParser(),        # Add in appropriate priority order
            DeepSeekXMLParser(),
            StandardXMLParser(),
            # ...
        ]
```

3. **Write tests** in `tests/unit/test_tool_parsers.py`:

```python
class TestMyModelParser:
    def test_can_parse_mymodel_format(self):
        parser = MyModelParser()
        assert parser.can_parse("<mymodel>...</mymodel>")

    def test_parse_mymodel(self):
        parser = MyModelParser()
        calls, cleaned = parser.parse(content)
        assert len(calls) == 1
```

## Design Principles

1. **Single Responsibility**: Each parser handles one format
2. **Open/Closed**: Easy to add new parsers without modifying existing code
3. **Clean Code**: Well-documented, < 200 lines per file
4. **Testable**: Each parser can be tested independently
5. **Logging**: Debug logging shows which parser was used
6. **Type Safety**: Full type hints throughout

## Logging

The parsers log at appropriate levels:

```python
# DEBUG: Normal operation
logger.debug(f"[{self.name}] Parsed {len(tool_calls)} tool call(s)")

# INFO: Successful fallback to generic parser
logger.info(f"[GenericXMLParser] Parsed {len(tool_calls)} using fallback")

# WARNING: Content looks like tool calls but none parsed
logger.warning(f"[{self.name}] Content looks like it has tool calls but none parsed")
```

## Performance

- **Lightweight checks**: `can_parse()` uses simple regex checks
- **Lazy parsing**: Only the matching parser does full extraction
- **No redundant work**: Chain stops at first successful parse
- **Efficient regex**: Pre-compiled patterns, minimal backtracking

## Future Enhancements

Potential improvements:
- [ ] Support for nested parameters
- [ ] Streaming parser for very large outputs
- [ ] Parser selection based on model name
- [ ] JSON schema validation for parameters
- [ ] Metrics/telemetry for parser usage
