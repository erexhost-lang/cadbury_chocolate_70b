"""
Language Server Protocol (LSP) Compiler Tools for Surgical Agent Navigation
Equips Cadbury-70B with IDE-grade compiler navigation tools:
1. lsp_find_definition: Jumps directly to symbol definition (class/function/struct).
2. lsp_find_references: Discovers all call sites across the repository in 1 hop.
3. lsp_get_type_signature: Extracts compiler/type-checker parameter signatures without file reading.
4. lsp_get_diagnostics: Returns instant static analysis (pyright/mypy/tsc/rustc) error diagnostics.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


# ==============================================================================
# 1. Tool JSON Schemas (Model Function Calling Specs)
# ==============================================================================
LSP_TOOL_DEFINITIONS = [
    {
        "name": "lsp_find_definition",
        "description": "Locate the exact file and line number where a symbol (class, function, variable) is defined.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "The exact symbol identifier name"},
                "file_path": {"type": "string", "description": "Current file context"},
                "line": {"type": "integer", "description": "Line number where symbol is called"},
            },
            "required": ["symbol", "file_path"],
        },
    },
    {
        "name": "lsp_find_references",
        "description": "Find all call sites and references to a symbol across the entire repository in 1 hop.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "The symbol identifier to find references for"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "lsp_get_type_signature",
        "description": "Query compiler type checker for a function, class, or variable type signature.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "The symbol identifier"},
                "file_path": {"type": "string", "description": "File containing the symbol"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "lsp_get_diagnostics",
        "description": "Query language server for instant static analysis errors and linter warnings in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative file path to inspect for compiler diagnostics"},
            },
            "required": ["file_path"],
        },
    },
]


# ==============================================================================
# 2. Mock LSP Server Client (For Synthetic Data & Testing)
# ==============================================================================
class MockLSPClient:
    """Simulates response payloads from Pyright / Rust-Analyzer / TypeScript Language Servers."""

    def __init__(self, symbol_table: Optional[Dict[str, Any]] = None):
        self.symbol_table = symbol_table or {
            "User": {
                "def_file": "models/user.py",
                "def_line": 35,
                "type": "class User(Base)",
                "references": ["services/user_service.py:12", "api/routes/auth.py:44"],
            },
            "validate_token": {
                "def_file": "auth/token.py",
                "def_line": 20,
                "type": "def validate_token(token: str) -> Optional[TokenPayload]",
                "references": ["middleware/auth_middleware.py:38"],
            },
        }

    def find_definition(self, symbol: str, file_path: str, line: Optional[int] = None) -> Dict[str, Any]:
        info = self.symbol_table.get(symbol)
        if info:
            return {
                "status": "success",
                "file": info["def_file"],
                "line": info["def_line"],
                "snippet": info.get("type", ""),
            }
        return {"status": "error", "message": f"Symbol '{symbol}' not found in compiler index."}

    def find_references(self, symbol: str) -> Dict[str, Any]:
        info = self.symbol_table.get(symbol)
        if info:
            return {
                "status": "success",
                "symbol": symbol,
                "references_count": len(info.get("references", [])),
                "locations": info.get("references", []),
            }
        return {"status": "success", "symbol": symbol, "references_count": 0, "locations": []}

    def get_type_signature(self, symbol: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        info = self.symbol_table.get(symbol)
        if info:
            return {
                "status": "success",
                "symbol": symbol,
                "signature": info["type"],
            }
        return {"status": "error", "message": f"No type information available for '{symbol}'."}

    def get_diagnostics(self, file_path: str) -> Dict[str, Any]:
        """Simulates instant static compiler errors."""
        if "user.py" in file_path:
            return {
                "status": "success",
                "file": file_path,
                "diagnostics": [
                    {
                        "severity": "error",
                        "line": 39,
                        "message": "Attribute 'is_verified' is declared in SQLAlchemy table but missing type annotation.",
                        "rule": "reportGeneralTypeIssues",
                    }
                ],
            }
        return {"status": "success", "file": file_path, "diagnostics": []}


if __name__ == "__main__":
    print("--- Testing LSP Compiler Tools Locally ---")
    lsp = MockLSPClient()

    print("1. Find Definition of 'User':")
    print(json.dumps(lsp.find_definition("User", "api/routes/auth.py"), indent=2))

    print("\n2. Find References of 'validate_token':")
    print(json.dumps(lsp.find_references("validate_token"), indent=2))

    print("\n3. Get Diagnostics for 'models/user.py':")
    print(json.dumps(lsp.get_diagnostics("models/user.py"), indent=2))
