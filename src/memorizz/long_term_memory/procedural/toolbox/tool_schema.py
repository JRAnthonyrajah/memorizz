from __future__ import annotations
from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator

# ---- Canonical, limited set of primitive types ----
class ParamType(str, Enum):
    string  = "string"
    integer = "integer"
    number  = "number"
    boolean = "boolean"
    array   = "array"
    object  = "object"

# ---- Simple items schema for arrays (extend if you need objects/properties) ----
class ItemsSchema(BaseModel):
    type: Literal["string", "integer", "number", "boolean", "object"]

class ParameterSchema(BaseModel):
    """
    Flat parameter row for metadata UIs and query generators.
    If type == 'array', 'items' MUST be present (auto-filled to string if omitted).
    """
    name: str
    description: str = ""
    type: ParamType
    required: bool = False
    items: Optional[ItemsSchema] = None  # only meaningful when type=='array'

    # --- Normalize common aliases BEFORE type is parsed into Enum ---
    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type_aliases(cls, v):
        if v is None:
            return v
        s = str(v).strip().lower()
        # accept a few common authoring shortcuts
        if s in {"array of strings", "string[]", "list[str]", "list of strings"}:
            return "array"
        return s

    # If callers provide ad-hoc "items_type" fields, honor them
    @model_validator(mode="before")
    @classmethod
    def _fold_items_type(cls, data: dict):
        if not isinstance(data, dict):
            return data
        if data.get("type") in ("array", ParamType.array):
            it = data.get("items")
            if it is None:
                # allow items_type: "string" etc.
                itype = str(data.get("items_type") or "").strip().lower() or "string"
                data["items"] = {"type": itype}
        else:
            # Not an array: drop stray items/items_type
            data.pop("items", None)
            data.pop("items_type", None)
        return data

    # Enforce items presence for arrays; remove items for non-arrays
    @model_validator(mode="after")
    def _enforce_array_items(self):
        if self.type == ParamType.array:
            if self.items is None:
                # policy: default to string (instead of raising)
                self.items = ItemsSchema(type="string")
        else:
            self.items = None
        return self

class FunctionSchema(BaseModel):
    """
    Metadata for a callable. 'parameters' is your flat list for UIs/prompts.
    'required' must reference names present in 'parameters'.
    """
    name: str
    description: str
    parameters: List[ParameterSchema]
    required: List[str] = []
    queries: List[str] = []

    @model_validator(mode="after")
    def _validate_required_subset(self):
        names = {p.name for p in self.parameters}
        missing = [r for r in self.required if r not in names]
        if missing:
            # choose policy: raise or silently drop. Here we raise to keep metadata sane.
            raise ValueError(f"'required' contains unknown parameter(s): {missing}")
        return self

class ToolSchemaType(BaseModel):
    """
    Container for a tool definition (e.g., OpenAI/Google function metadata wrapper).
    """
    type: str  # typically 'function'
    function: FunctionSchema
