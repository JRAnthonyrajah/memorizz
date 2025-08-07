from pydantic import BaseModel
from typing import List

class ParameterSchema(BaseModel):
    """
    A schema for the parameter.
    """
    name: str
    description: str
    type: str
    required: bool

class FunctionSchema(BaseModel):
    """
    A schema for the function.
    """
    name: str
    description: str
    parameters: list[ParameterSchema]
    required: List[str]
    queries: List[str]
    
class ToolSchemaType(BaseModel):
    """
    A schema for the tool.
    This can be the OpenAI function calling schema or Google function calling schema.
    """
    type: str
    function: FunctionSchema