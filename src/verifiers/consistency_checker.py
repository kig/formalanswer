import re
from typing import Dict, List, Tuple

def extract_tla_constants(content: str) -> Dict[str, str]:
    """
    Extracts defined constants from TLA+ content (Simple values only).
    Matches:
      - MaxRetries == 5
      - Enabled == TRUE
      - Name == "Test"
    Returns: Dict[Name, Value]
    """
    constants = {}
    
    # Match definitions: Name == Value
    # Value must be simple: Number, TRUE/FALSE, String
    def_pattern = re.compile(r'^\s*(\w+)\s*==\s*([0-9]+|TRUE|FALSE|"[^"]*")', re.MULTILINE)
    
    for match in def_pattern.finditer(content):
        name = match.group(1)
        val = match.group(2).strip()
        constants[name] = val
        
    return constants

def extract_python_constants(content: str) -> Dict[str, str]:
    """
    Extracts global constant assignments from Python content (Simple values only).
    Matches:
      - MAX_RETRIES = 5
      - ENABLED = True
      - NAME = "Test"
    """
    constants = {}
    
    # Regex: CONSTANT_NAME = Value
    # Value must be simple: Number, True/False, String
    assign_pattern = re.compile(r'^\s*([A-Z][A-Z0-9_]*|[A-Z][a-zA-Z0-9]*)\s*=\s*([0-9]+|True|False|"[^"]*"|\'[^\']*\')', re.MULTILINE)
    
    for match in assign_pattern.finditer(content):
        name = match.group(1)
        val = match.group(2).strip()
        constants[name] = val
        
    return constants

def normalize_name(name: str) -> str:
    """
    Normalizes variable names for comparison.
    MAX_RETRIES -> maxretries
    MaxRetries -> maxretries
    """
    return name.lower().replace("_", "")

def check_consistency(tla_content: str, python_content: str) -> List[str]:
    """
    Checks if constants in TLA+ and Python match.
    Returns a list of warnings.
    """
    warnings = []
    
    tla_consts = extract_tla_constants(tla_content)
    py_consts = extract_python_constants(python_content)
    
    # Map normalized names back to original names and values
    tla_norm = {normalize_name(k): (k, v) for k, v in tla_consts.items()}
    py_norm = {normalize_name(k): (k, v) for k, v in py_consts.items()}
    
    # Check intersection
    common_keys = set(tla_norm.keys()) & set(py_norm.keys())
    
    for key in common_keys:
        tla_name, tla_val = tla_norm[key]
        py_name, py_val = py_norm[key]
        
        # Strip quotes for strings
        tla_clean = tla_val.strip('"')
        py_clean = py_val.strip('"\'')
        
        # Normalize Booleans (TRUE -> True)
        if tla_clean == "TRUE": tla_clean = "True"
        if tla_clean == "FALSE": tla_clean = "False"
        
        if tla_clean != py_clean:
            # Try parsing as float/int
            try:
                if float(tla_clean) == float(py_clean):
                    continue
            except ValueError:
                pass
                
            warnings.append(f"Consistency Warning: TLA+ defines '{tla_name}' as '{tla_val}', but Python defines '{py_name}' as '{py_val}'.")
            
    return warnings