import argparse
import sys
from src.controller import FormalReasoningLoop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Formal Reasoning Loop")
    parser.add_argument("task", nargs="*", help="The natural language query/task (can be multiple strings)")
    parser.add_argument("--prompt-file", help="Load the prompt from a file.")
    parser.add_argument("--backend", default="gemini", choices=["gemini", "openai", "ollama", "mock"], help="LLM backend to use")
    parser.add_argument("--model", help="Specific model name (e.g., gpt-4, gemini-2.5-flash, llama3)")
    parser.add_argument("--base-url", help="Base URL for OpenAI/Ollama API")
    parser.add_argument("--api-key", help="API Key (overrides env vars)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed verification errors in output")
    parser.add_argument("--combat", action="store_true", help="Enable Adversarial Combat Mode (Red Team review)")
    parser.add_argument("--peer-review", action="store_true", help="Enable Constructive Peer Review Mode")
    parser.add_argument("--rap-battle", action="store_true", help="Enable Logic Rap Battle Mode")
    parser.add_argument("--mode", choices=["discrete", "probabilistic", "hybrid", "factual"], help="Force a specific reasoning mode")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of reasoning iterations")
    parser.add_argument("--construct-rap", nargs='?', const='CURRENT', help="Construct rap lyrics from history. Provide directory path for existing task, or flag for current session.")
    
    args = parser.parse_args()
    
    if args.construct_rap and args.construct_rap != 'CURRENT':
        # Standalone mode: Generate rap from existing directory
        frl = FormalReasoningLoop(
            backend=args.backend,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url
        )
        frl.finalize_rap_battle(args.construct_rap)
        sys.exit(0)

    task_parts = []
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            task_parts.append(f.read())
    if args.task:
        task_parts.append(" ".join(args.task))
        
    if not task_parts:
        task = "How could AI agents use formal methods to produce superhuman thinking?"
    else:
        task = "\n\n".join(task_parts)

    frl = FormalReasoningLoop(
        max_iterations=args.max_iterations,
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        verbose=args.verbose,
        combat=args.combat,
        peer_review=args.peer_review,
        rap_battle=args.rap_battle,
        generate_rap=(args.construct_rap == 'CURRENT'),
        force_mode=args.mode
    )
    frl.run(task)