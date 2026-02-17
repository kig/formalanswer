import ast
import yaml
import re

def inject_runtime_checks(code: str, skill_defs: list) -> str:
    """
    Wraps known skill functions with runtime assertion checks.
    """
    wrappers = []
    
    for skill_path in skill_defs:
        with open(skill_path, 'r') as f:
            skill = yaml.safe_load(f)
            
        name = skill['name']
        inputs = skill['inputs']
        pre = skill.get('preconditions', [])
        post = skill.get('postconditions', [])
        
        args = ", ".join(inputs.keys())
        
        # Construct wrapper function
        wrapper_lines = []
        wrapper_lines.append(f"def verified_{name}({args}):")
        
        # Preconditions
        for p in pre:
            wrapper_lines.append(f"    assert {p}, 'Precondition failed: {p}'")
            
        # Original Call
        wrapper_lines.append(f"    result = {name}({args})")
        
        # Postconditions
        for p in post:
            wrapper_lines.append(f"    assert {p}, 'Postcondition failed: {p}'")
            
        wrapper_lines.append("    return result")
        
        wrappers.append("\n".join(wrapper_lines))
        
        # Replace calls in code using regex
        # Pattern: \bname\(
        pattern = r'\b' + name + r'\('
        replacement = f'verified_{name}('
        code = re.sub(pattern, replacement, code)
        
    return "\n".join(wrappers) + "\n\n" + code
