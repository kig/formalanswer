import re

def extract_lean_interface(content: str) -> str:
    """
    Extracts the public interface (theorems, definitions, axioms) from a Lean 4 file.
    Strips proofs (blocks starting with ':= by' or ':= proof').
    """
    lines = content.split('\n')
    interface_lines = []
    
    # Regex for standard declarations
    # Matches: theorem name (args) : type
    decl_pattern = re.compile(r'^\s*(theorem|def|axiom|constant|structure|class)\s+.*')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Check if it's a declaration
        if decl_pattern.match(stripped):
            # If line contains ':=', cut it off
            if ':=' in line:
                head, _ = line.split(':=', 1)
                interface_lines.append(head.strip())
            else:
                # Assuming multi-line signature or type declaration
                interface_lines.append(line.strip())
        elif stripped.startswith("--"): 
            # Keep top-level comments for context
            interface_lines.append(line)
            
    return "\n".join(interface_lines)

def extract_tla_interface(content: str) -> str:
    """
    Extracts constants, variables, and operator signatures from TLA+ files.
    """
    lines = content.split('\n')
    interface_lines = []
    
    # Regex for TLA+ declarations
    decl_pattern = re.compile(r'^\s*(CONSTANT|VARIABLE|.*==.*)')
    module_pattern = re.compile(r'^\s*(----|====|EXTENDS|MODULE)')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        if module_pattern.match(stripped):
            interface_lines.append(line)
        elif decl_pattern.match(stripped):
            # Keep full definition for TLA+ (operators are usually short)
            # But stripped of complex logic if possible? 
            # For TLA+, the definitions ARE the logic, so we keep them 
            # unless we implement a sophisticated parser.
            # A simple heuristic: Keep header, maybe truncate long bodies?
            # For now, let's keep full definitions as TLA modules are usually small.
            interface_lines.append(line)
        elif stripped.startswith("\\*"):
            interface_lines.append(line)
            
    return "\n".join(interface_lines)