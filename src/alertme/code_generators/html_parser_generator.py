import os
import logging
from dotenv import load_dotenv
from alertme.utils.codegen import CodeGenerator, combined_code

def generate_html_parser():
    from alertme.web_voyager.tasks.tennis_task import get_html_parser_test_cases, get_html_parser_code_gen_prompt
    
    load_dotenv()

    test_cases = get_html_parser_test_cases()    
    codegen_prompt = get_html_parser_code_gen_prompt(test_cases)    
    code_generator = CodeGenerator()    
    code_solution = code_generator.generate_code(codegen_prompt, test_cases)
    result = combined_code(code_solution)

    return result

def save_html_parser(result):
    os.makedirs('data', exist_ok=True)

    with open('data/html_parser_func.py', 'w') as f:
        f.write(result)

    logging.info("Code has been written to data/html_parser_func.py")
    
def read_html_from_file(filename = 'web_voyager/data/html_result.txt'):
    with open(filename, 'r') as f:
        return f.read()

if __name__ == "__main__":
    result = generate_html_parser()
    print("RESULT:\n\n" + str(result) + "\n\n")
    save_html_parser(result)