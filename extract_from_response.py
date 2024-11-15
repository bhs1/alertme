from codegen import CodeGenerator, combined_code, exec_as_main
import os

def generate_response_func(test_cases):
    os.environ["LANGCHAIN_PROJECT"] = "generate_response_func"
    USER_PROMPT = f"""Write a function to extract and list all available playing times 
                from HTTP response content.
                
                Example HTTP response content:
                {test_cases[0]['inputs']}
                
                Example output:
                {test_cases[0]['outputs']}
            """

    code_solution = CodeGenerator().generate_code(USER_PROMPT, test_cases)
    print("RESULT:\n\n" + str(combined_code(code_solution)) + "\n\n")
    return combined_code(code_solution)

def extract_info_from_response(response_content, response_func):
    print("====================== Executing response function ========================\n", response_func)
    print("====================== Response content top 1000 ========================\n", response_content[:1000])
    global_scope = {'global_input': response_content,
                    'global_output': ""}
    exec_as_main(response_func, global_scope)
    return global_scope['global_output']
