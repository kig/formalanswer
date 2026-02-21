from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.markdown import Markdown
from contextlib import contextmanager

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "success": "bold green",
    "panel.border": "blue"
})

console = Console(theme=custom_theme)

def log_info(msg):
    console.print(f"[info]{msg}[/info]")

def log_warning(msg):
    console.print(f"[warning]{msg}[/warning]")

def log_success(msg):
    console.print(f"[success]{msg}[/success]")

def log_error(msg):
    console.print(f"[error]{msg}[/error]")

def log_section(title, content, style="blue"):
    console.print(Panel(Markdown(content), title=title, border_style=style))

def print_markdown(content):
    console.print(Markdown(content))

@contextmanager
def status(msg):
    with console.status(msg, spinner="dots") as s:
        yield s
