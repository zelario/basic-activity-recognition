from datetime import datetime

def print_and_log(*args, log_file_path="log/output.log", sep=" ", end="\n"):
    """
    Prints to terminal and appends the message to a log file.
    Parameters:
        *args: Values to print.
        log_file_path: Path to the log file (default: "../log/output.log").
        sep: Separator between values (default: space).
        end: End character (default: newline).
    """
    message = sep.join(str(arg) for arg in args) + end
    print(message, end="")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(message)
