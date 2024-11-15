import sys
sys.path.insert(0, '.')
sys.path.insert(0, './web_voyager/')


from drivergen import CodeGenerator
from drivergen import combined_code
import os
import logging
from dotenv import load_dotenv
from utils import format_to_yaml


def get_generalize_code_prompt(test_cases, starter_code, task_prompt, playwright_actions_code_lib):        
    os.environ["LANGCHAIN_PROJECT"] = "codegen"
    
    # Format the first test case to YAML
    formatted_test_case = format_to_yaml(test_cases[0])
    
    # Define the prompt for the LLM
    prompt = f"""An AI has written python code to complete this task:
    
    {task_prompt}
    
The starter code is:
    
    {starter_code}
    
your task is to generalize the the code so that it passes all test cases. Here is an example of the first test case:
    
{formatted_test_case}
    
Instead of using typical playwright actions, use the playwright_actions_core.py functions:
    
    {playwright_actions_code_lib}
    
IMPORTANT: Don't remove any steps from the starter code. Your main task is just to generalize
any steps that are hardcoded which causes test failures. But even steps that are considered a failure
in the comment can be beneficial and removing a step can be detrimental.
    """
    return prompt

def get_playwright_actions_code_lib():
    with open("web_voyager/playwright_actions_core.py", "r") as file:
        playwright_actions_code_lib = file.read()
    return playwright_actions_code_lib


if __name__ == "__main__":
    from web_voyager.tasks.tennis_task import get_prompt, get_ai_generated_start_code, get_task_params, get_html_parser, custom_code_task_guideance    
    load_dotenv()
    task_params = get_task_params()
    expected_output = {
        "Available court times": {'Tennis': ['10:00am', '10:30am', '11:00am', '11:30am', '12:00pm', '12:30pm', '1:00pm', '1:30pm', '2:00pm', '2:30pm', '3:00pm', '3:30pm', '5:00pm', '5:30pm']}
    }
    test_cases = [{'inputs': task_params, 'outputs': expected_output}]
    playwright_actions_code_lib = get_playwright_actions_code_lib()
    codegen_prompt = get_generalize_code_prompt(test_cases, get_ai_generated_start_code(), get_prompt(), playwright_actions_code_lib)
    extra_task_guideance = custom_code_task_guideance()
    codegen_prompt += extra_task_guideance
    code_generator = CodeGenerator(
        codegen_prompt, start_at_condense=False, use_cached_navigation_code=True, html_parser=get_html_parser())
    code_solution = code_generator.generate_code(test_cases)
    print("RESULT:\n\n" + str(combined_code(code_solution)) + "\n\n")

    result = combined_code(code_solution)

    os.makedirs('data', exist_ok=True)

    with open('data/activity_times_func.py', 'w') as f:
        f.write(result)

    logging.info("Code has been written to data/activity_times_func.py")

    code_generator.close()