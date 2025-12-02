from datetime import datetime

def print_and_log(*args, path="log/output.log", sep=" ", end="\n"):
    """
    Prints to terminal and appends the message to a log file.
    Parameters:
        *args: Values to print.
        log_file_path: Path to the log file (default: "../log/output.log").
        sep: Separator between values (default: space).
        end: End character (default: newline)."""
    
    message = sep.join(str(arg) for arg in args) + end
    print(message, end="")
    with open(path, "a", encoding="utf-8") as f:
        f.write(message)

def clear_and_print(*args, sep=" ", end="\n"):
    """
    Prints to terminal and appends the message to a log file.
    Parameters:
        *args: Values to print.
        log_file_path: Path to the log file (default: "../log/output.log").
        sep: Separator between values (default: space).
        end: End character (default: newline)."""
    
    message = sep.join(str(arg) for arg in args) + end
    print(message, end="")
    with open("log/output.log", "w", encoding="utf-8") as f:
        f.write(message)
    with open("log/hyperparameter_tuning.log", "w", encoding="utf-8") as f:
        f.write(message)
    with open("log/validation_metrics.log", "w", encoding="utf-8") as f:
        f.write(message)
