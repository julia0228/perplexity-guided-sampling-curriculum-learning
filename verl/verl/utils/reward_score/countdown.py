import re
import random
import ast
import operator
from verl.utils.reward_score.utils import remove_boxed, last_boxed_only_string, extract_solution

# def extract_solution(solution_str):
#     """Extract the equation from the solution string."""
#     # Remove everything before the first "Assistant:"
#     if "Assistant:" in solution_str:
#         solution_str = solution_str.split("Assistant:", 1)[1]
#     elif "<|im_start|>assistant" in solution_str:
#         solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
#     else:
#         return None
#     solution_str = solution_str.split('\n')[-1]

#     answer_pattern = r'<answer>(.*?)</answer>'
#     match = re.finditer(answer_pattern, solution_str)
#     matches = list(match)
#     if matches:
#         final_answer = matches[-1].group(1).strip()
#     else:
#         final_answer = None
#     return final_answer



def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Strip the "= target" part if present (e.g., "50 + 46 - 35 = 61" -> "50 + 46 - 35")
        equation_only = re.split(r'\s*=\s*', equation_str)[0]

        # Extract all numbers from the expression (left side of =)
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_only)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions.

    Handles both "2 * 5 - 3" and "2 * 5 - 3 = 7" formats.
    If "= answer" is present, verify it matches the computed result.
    """
    try:
        parts = re.split(r'\s*=\s*', equation_str)
        expr = parts[0].strip()

        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, expr):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the expression
        result = eval(expr, {"__builtins__": None}, {})

        # If "= answer" is present, verify the stated answer matches the computed result
        if len(parts) > 1:
            try:
                stated_answer = float(parts[1].strip())
                if abs(result - stated_answer) > 1e-5:
                    return None  # Stated answer doesn't match computed result
            except ValueError:
                pass  # Non-numeric answer after =, just use computed result

        return result
    except Exception as e:
        return None



def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.

    Scoring:
    - No \\boxed{} found: 0
    - \\boxed{} found but numbers wrong: 0
    - Numbers match exactly (correct count & values): 0.1
    - Numbers match + valid evaluable format: 0.2
    - Numbers match + result == target: 1.0
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    numbers = [int(n) for n in numbers.split(',')]

    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Check if numbers match exactly
    numbers_match = validate_equation(equation, numbers)

    if not numbers_match:
        if do_print:
            print(f"Numbers mismatch")
        return 0

    # Numbers match — at least 0.1
    # Try to evaluate the expression
    try:
        result = evaluate_equation(equation)
        if result is None:
            # Numbers match but format not evaluable
            if do_print:
                print(f"Numbers match but cannot evaluate")
            return 0.1

        # Numbers match + format valid = 0.2
        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Numbers match, format valid, wrong result: {result} != {target}")
            return 0.2
    except:
        if do_print:
            print(f"Numbers match but evaluation error")
        return 0.1 


def get_countdown_compute_score(correct_score, format_score):
    return lambda solution_str, ground_truth: compute_score(solution_str, ground_truth, method='strict', format_score=format_score, score=correct_score)
