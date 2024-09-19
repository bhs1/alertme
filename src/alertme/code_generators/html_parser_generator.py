import logging

from dotenv import load_dotenv

from alertme.utils.codegen import CodeGenerator, combined_code
from alertme.utils.path_utils import get_output_path
from alertme.web_voyager.tasks.tennis_task import get_html_parser_test_cases, get_html_parser_code_gen_prompt


def generate_html_parser():

    test_cases = get_html_parser_test_cases()
    codegen_prompt = get_html_parser_code_gen_prompt(test_cases)
    code_generator = CodeGenerator()
    code_solution = code_generator.generate_code(codegen_prompt, test_cases)
    result = combined_code(code_solution)

    return result


def save_html_parser(result):
    output_file = get_output_path() / 'html_parser_func.py'
    with open(output_file, 'w') as f:
        f.write(result)

    logging.info(f"Code has been written to {output_file}")


def read_html_from_file(filename):
    with open(filename, 'r') as f:
        return f.read()


if __name__ == "__main__":
    load_dotenv()
    result = generate_html_parser()
    print("RESULT:\n\n" + str(result) + "\n\n")
    save_html_parser(result)
