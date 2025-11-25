"""Graph data models."""

from typing import List
from pydantic import BaseModel, Field

class GraphNode(BaseModel):
    """Represents a single note in the graph."""
    id: str = Field(..., description="Unique identifier (Note Path)")
    label: str = Field(..., description="Display title of the note")
    val: int = Field(default=1, description="Weight/Size of the node")
    group: str = Field(..., description="Grouping category (e.g., top-level folder)")

class GraphLink(BaseModel):
    """Represents a directed connection between two notes."""
    source: str = Field(..., description="ID of the source note")
    target: str = Field(..., description="ID of the target note")

class GraphData(BaseModel):
    """The top-level payload returned by the API."""
    nodes: List[GraphNode]
    links: List[GraphLink]
