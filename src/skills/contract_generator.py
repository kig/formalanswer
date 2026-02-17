import yaml
import os

def generate_contracts(skill_path):
    with open(skill_path, 'r') as f:
        skill = yaml.safe_load(f)
        
    name = skill['name']
    inputs = skill['inputs']
    preconditions = skill.get('preconditions', [])
    postconditions = skill.get('postconditions', [])
    
    lean_contract = _generate_lean(name, inputs, preconditions, postconditions)
    tla_contract = _generate_tla(name, inputs, preconditions, postconditions)
    
    return lean_contract, tla_contract

def _generate_lean(name, inputs, pre, post):
    # Map YAML types to Lean types
    type_map = {'Int': 'Int', 'String': 'String', 'Bool': 'Bool'}
    
    args = " ".join([f"({k} : {type_map.get(v, 'Any')})" for k, v in inputs.items()])
    
    lines = []
    lines.append(f"-- Contract for {name}")
    lines.append(f"constant {name} {args} : IO Int") # Assuming Int result for now
    
    # Generate Axiom
    # For simplicity, join conditions with ∧
    pre_str = " ∧ ".join(pre) if pre else "True"
    post_str = " ∧ ".join(post) if post else "True"
    
    lines.append(f"axiom {name}_spec {args} :")
    lines.append(f"  ({pre_str}) → ({post_str})")
    
    return "\n".join(lines)

def _generate_tla(name, inputs, pre, post):
    args = ", ".join(inputs.keys())
    
    lines = []
    lines.append(f"(* Contract for {name} *)")
    
    # Precondition Operator
    pre_str = " /\\ ".join(pre) if pre else "TRUE"
    # Replace python/lean operators with TLA+ equivalents if needed (e.g. == is =)
    # This is a naive replacement, robust parser needed for complex logic
    pre_str = pre_str.replace("==", "=").replace("&&", "/\\").replace("||", "\\/")
    
    lines.append(f"{name}Pre({args}) == {pre_str}")
    
    # Postcondition Operator (requires result arg)
    post_str = " /\\ ".join(post) if post else "TRUE"
    post_str = post_str.replace("==", "=").replace("&&", "/\\").replace("||", "\\/")
    
    lines.append(f"{name}Post({args}, result) == {post_str}")
    
    return "\n".join(lines)