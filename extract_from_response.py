from codegen import CodeGenerator, combined_code

def generate_response_func(test_cases):
    # Find a few more interesting cases.
    # If they fail or take many iterations, try with a debug node.
    USER_PROMPT = f"""Write a function to extract and list all available playing times 
                from HTTP response content.
                
                Example HTTP response content:
                {test_cases[0]['inputs']}
                
                Example output:
                {test_cases[0]['outputs']}

                If there are multiple activity types, use a dictionary instead of a list and associate times
                with the correct activity.            

                Please pay attention to the system message above to see how to format your response.
                """

    code_solution = CodeGenerator().generate_code(USER_PROMPT, test_cases)
    print("RESULT:\n\n" + str(code_solution) + "\n\n")
    return combined_code(code_solution)

def extract_info_from_response(response_content, response_func):
    print("====================== Executing response function ========================\n", response_func)
    print("====================== Response content top 1000 ========================\n", response_content[:1000])
    global_scope = {'global_input': response_content,
                    'global_output': ""}
    exec(response_func, global_scope)
    return global_scope['global_output']
