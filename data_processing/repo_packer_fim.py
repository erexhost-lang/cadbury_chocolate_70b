"""
Repository-Level Topological Dependency Packing & Fill-In-The-Middle (FIM)
1. Parses AST imports to construct an intra-repository Directed Acyclic Graph (DAG).
2. Topologically sorts files (e.g. types/models -> repositories -> services -> controllers)
   so foundational definitions appear in the transformer context BEFORE dependent calls.
3. Applies probabilistic FIM (Prefix-Suffix-Middle / Suffix-Prefix-Middle) so the model
   learns middle-of-file code insertion alongside standard left-to-right generation.
"""

import ast
import os
import random
import sys
from typing import Dict, List, Set, Tuple, Optional, Generator
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import setup_logger

logger = setup_logger("repo_packer_fim")

# Canonical FIM Token Tags
FIM_PRE = "<|fim_prefix|>"
FIM_SUF = "<|fim_suffix|>"
FIM_MID = "<|fim_middle|>"
DOC_SEP = "<|doc|>"


class RepoDependencyParser:
    """Parses AST import statements across a Python repository to construct an import graph."""

    @staticmethod
    def extract_imports_from_code(code: str) -> List[str]:
        """Extract imported module names using Python's native AST."""
        imported_modules = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_modules.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.append(node.module)
        except Exception:
            # Fallback regex/token scanning if AST fails
            for line in code.split("\n"):
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    parts = line.split()
                    if len(parts) >= 2:
                        imported_modules.append(parts[1].split(".")[0])
        return imported_modules

    @classmethod
    def build_repo_dependency_graph(cls, repo_files: Dict[str, str]) -> List[str]:
        """
        Builds a directed dependency graph: A -> B indicates file A depends on file B.
        Returns a topologically sorted list where independent files (models, utils) appear first.
        """
        graph = nx.DiGraph()
        
        # Add all files as nodes
        for file_path in repo_files.keys():
            graph.add_node(file_path)

        # Normalize paths to module dot-notation (e.g. 'models/user.py' -> 'models.user')
        path_to_mod = {}
        for path in repo_files.keys():
            mod_name = os.path.splitext(path)[0].replace("/", ".").replace("\\", ".")
            path_to_mod[mod_name] = path

        # Add directed edges: if file A imports file B, add edge B -> A
        # (So B is evaluated before A in topological sort)
        for path, code in repo_files.items():
            imported_mods = cls.extract_imports_from_code(code)
            for imp in imported_mods:
                # Check for direct or prefix matches in repo
                for mod_name, target_path in path_to_mod.items():
                    if imp == mod_name or imp.startswith(mod_name + "."):
                        if target_path != path:
                            # target_path must come BEFORE path
                            graph.add_edge(target_path, path)

        # Handle and break cyclic dependencies (e.g. circular imports)
        while not nx.is_directed_acyclic_graph(graph):
            cycle = nx.find_cycle(graph, orientation="original")
            # Break cycle by removing the weakest feedback edge
            graph.remove_edge(cycle[0][0], cycle[0][1])

        # Topologically sort nodes
        sorted_files = list(nx.topological_sort(graph))
        return sorted_files


def apply_fim_slice(
    code: str,
    fim_rate: float = 0.5,
    spm_rate: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[str, bool]:
    """
    Applies Fill-In-The-Middle transformation.
    PSM: <|fim_prefix|> Prefix <|fim_suffix|> Suffix <|fim_middle|> Middle
    SPM: <|fim_prefix|> Suffix <|fim_suffix|> Prefix <|fim_middle|> Middle
    """
    if seed is not None:
        random.seed(seed)

    if random.random() > fim_rate or len(code) < 40:
        return code, False

    # Pick two boundary cut indices
    idx1 = random.randint(0, len(code))
    idx2 = random.randint(0, len(code))
    start_cut, end_cut = sorted([idx1, idx2])

    prefix = code[:start_cut]
    middle = code[start_cut:end_cut]
    suffix = code[end_cut:]

    # Decide ordering: Prefix-Suffix-Middle (PSM) vs Suffix-Prefix-Middle (SPM)
    if random.random() < spm_rate:
        # SPM
        transformed = f"{FIM_PRE}{suffix}{FIM_SUF}{prefix}{FIM_MID}{middle}"
    else:
        # PSM
        transformed = f"{FIM_PRE}{prefix}{FIM_SUF}{suffix}{FIM_MID}{middle}"

    return transformed, True


class RepoPacker:
    """Concatenates topologically sorted repo files and packs them into long context windows."""

    def __init__(self, max_context_tokens: int = 32768, fim_rate: float = 0.5):
        self.max_context_tokens = max_context_tokens
        self.fim_rate = fim_rate

    def pack_repository(
        self,
        repo_files: Dict[str, str],
        tokenizer=None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Sorts repository by dependencies, applies FIM with probability `fim_rate`,
        and yields packed context sequences.
        """
        # 1. Topologically sort repository files
        sorted_paths = RepoDependencyParser.build_repo_dependency_graph(repo_files)
        logger.info(f"Topologically ordered {len(sorted_paths)} repository files:")
        for idx, p in enumerate(sorted_paths):
            logger.debug(f"  [{idx+1}/{len(sorted_paths)}] {p}")

        # 2. Assemble packed documents with explicit file boundaries
        current_pack = []
        current_length_approx = 0

        for path in sorted_paths:
            raw_code = repo_files[path]
            # Format file with header
            header = f"# ========================================================\n# File: {path}\n# ========================================================\n"
            processed_code, is_fim = apply_fim_slice(raw_code, fim_rate=self.fim_rate)
            
            doc_text = f"{header}{processed_code}\n\n{DOC_SEP}\n\n"
            # Approximate token count (char / 3.8 for code)
            est_tokens = int(len(doc_text) / 3.8)

            if current_length_approx + est_tokens > self.max_context_tokens and current_pack:
                yield {
                    "text": "".join(current_pack),
                    "estimated_tokens": current_length_approx,
                    "is_repo_packed": True,
                }
                current_pack = []
                current_length_approx = 0

            current_pack.append(doc_text)
            current_length_approx += est_tokens

        if current_pack:
            yield {
                "text": "".join(current_pack),
                "estimated_tokens": current_length_approx,
                "is_repo_packed": True,
            }


if __name__ == "__main__":
    # Demonstration on simulated micro-repository
    simulated_repo = {
        "services/order_service.py": """from models.order import Order
from utils.logger import log_event

def process_order(order_id: str):
    log_event(f'Processing {order_id}')
    return Order(order_id)
""",
        "models/order.py": """from models.base import BaseModel

class Order(BaseModel):
    def __init__(self, order_id: str):
        self.order_id = order_id
""",
        "models/base.py": """class BaseModel:
    pass
""",
        "utils/logger.py": """def log_event(msg: str):
    print(f'[EVENT] {msg}')
""",
    }

    print("--- Calculating Topological Order ---")
    sorted_order = RepoDependencyParser.build_repo_dependency_graph(simulated_repo)
    print("Resolved Execution Order (Dependencies First):")
    for step, f in enumerate(sorted_order, 1):
        print(f"  {step}. {f}")

    print("\n--- Packing Repository with FIM ---")
    packer = RepoPacker(max_context_tokens=4096, fim_rate=0.5)
    for pack in packer.pack_repository(simulated_repo):
        print(f"Packed chunk length: ~{pack['estimated_tokens']} tokens")
