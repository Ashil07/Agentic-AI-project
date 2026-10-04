def calculate(expression):
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except Exception:
        return "Invalid expression"
