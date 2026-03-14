"""User alert / notification system."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def info(message: str, title: str = "Info") -> None:
    console.print(Panel(message, title=f"[cyan]{title}[/]", border_style="cyan"))


def success(message: str, title: str = "Done") -> None:
    console.print(Panel(message, title=f"[green]{title}[/]", border_style="green"))


def warning(message: str, title: str = "Warning") -> None:
    console.print(Panel(message, title=f"[yellow]{title}[/]", border_style="yellow"))


def error(message: str, title: str = "Error") -> None:
    console.print(Panel(message, title=f"[red]{title}[/]", border_style="red"))


def breakthrough(message: str) -> None:
    """Eye-catching alert for potential breakthroughs."""
    text = Text()
    text.append("⚡ BREAKTHROUGH DETECTED ⚡\n\n", style="bold magenta blink")
    text.append(message, style="bright_white")
    console.print(
        Panel(
            text,
            title="[bold magenta]🔬 RESEARCH ALERT 🔬[/]",
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )
    # Try to make a sound
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass


def progress_update(task: str, step: int, total: int, detail: str = "") -> None:
    pct = (step / total * 100) if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * step // total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    msg = f"[{bar}] {pct:.0f}%  ({step}/{total})"
    if detail:
        msg += f"\n{detail}"
    console.print(Panel(msg, title=f"[blue]{task}[/]", border_style="blue"))


def ask_confirmation(message: str) -> bool:
    """Ask the user for yes/no confirmation."""
    console.print(f"\n[yellow]⚠  {message}[/]")
    response = console.input("[bold]Proceed? (y/n): [/]").strip().lower()
    return response in ("y", "yes")
