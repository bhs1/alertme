from codegen import CodeGenerator, combined_code, CODE_SUFFIX
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os
import json



def process_request_test_cases(test_case_tuples, request_params_map):
    test_cases = [{'inputs': combine_request_data(
        request_params_map, request), 'outputs': expected_output} for request, expected_output in test_case_tuples]
    return test_cases

# TODO: I had to manually replace 06/25/2024 instead of 05%2F09%2F2024 since the llm could not figure out it had to use %2F instead of /.
def get_request_replace_func(code_generator, test_cases, request_params_map):
    print(test_cases)
    test_cases = process_request_test_cases(test_cases, request_params_map)
    print(test_cases)
    os.environ["LANGCHAIN_PROJECT"] = "request_replace_func"
    # Define the prompt for the LLM
    prompt = f"""Write a Python function called "func" that takes a single str param which contains
    a json map. The map contains an HTTP request string and a dictionary where keys are field names and
    values are the new values for these fields. The function should return the updated HTTP request string.
    Here is an example test input {test_cases[0]['inputs']}.
    
    The request_params_map will be the first element of the dictionary, it will be used to decide which fields to
    replace in the http request. The request_data will be the second item of the dictionary. IMPORTANT: the func output should be
    the updated request_data WITHOUT the request_params_map.

    You may need to use regex to find the fields you are looking for. You may also need to translate the inputs
    into the correct format for the fields.
    
    Also, make sure you match the exact format, encoding, and styles in the expected output. It may not be sufficient to blindly
    replace with the values given in the request_params_map.

    """

    code_solution = code_generator.generate_code(prompt, test_cases)
    print("RESULT:\n\n" + str(combined_code(code_solution)) + "\n\n")

    return combined_code(code_solution)


class FieldMap(BaseModel):
    field_list: List[str] = Field(
        description="""A list of field names that the user would likely want to customize
        in order to search for avaialble playing times.""")
    value_list: List[str] = Field(
        description="""A list of values in the HTTP request that correspond to the fields in the field_list.
        The values should be in the same order as the fields in the field_list.""")


def get_request_params_map(request):
    os.environ["LANGCHAIN_PROJECT"] = "request_param_map"
    # Returns a dictionary with the field names that can be optionally customized by the user.
    prompt = f"""Given this HTTP request {request}, create the corresponding field_list and value_list.
    The fields should only be related to booking activity times. Ignore techincal fields like Host, User-Agent, IDs,
    and things that may not be iterpretable by humans. Try to keep the list at most 10 fields and sort in order of relevance.
    The values should be in the same order as the fields in the field_list. """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    field_map_chain = llm.with_structured_output(FieldMap, include_raw=False)
    field_map = field_map_chain.invoke(prompt)
    request_params_map = {field: value for field, value in zip(
        field_map.field_list, field_map.value_list)}
    return request_params_map

# Works for either header or body.
# TODO: Move the global_output, global_input stuff outside of the function that the LLM writes.


def replace_fields(request, request_params_map, func):
    """
    Executes a function to replace fields in an HTTP request using a combined request parameters map and request data.

    Args:
        request (dict): Original request data.
        request_params_map (dict): Parameters to prepend.
        func (str): Function string to replace fields in the HTTP request.

    Returns:
        str: The updated HTTP request after field replacement.
    """
    
    # Combine the request data and parameters
    combined_data = combine_request_data(request_params_map, request)

    # Prepare the execution environment
    args = {
        "global_input": combined_data,
        "global_output": "",
        "debug_output": ""
    }

    # Execute the function
    exec(func, args)    
    print(f"DEBUG: debug_output: {args['debug_output']}")
    return args["global_output"]


def combine_request_data(request_params_map, request):
    """
    Combines request parameters map and request data into a single dictionary and validates the combined JSON.

    Args:
        request_params_map (dict): Parameters to prepend.
        request (dict): Original request data.

    Returns:
        dict: Combined request data if valid, None otherwise.
    """
    # Ensure both inputs are dictionaries
    if not isinstance(request_params_map, dict) or not isinstance(request, dict):
        print("Error: Both request_params_map and request must be dictionaries")
        return None

    combined_data = {
        "request_params_map": request_params_map,
        "request_data": request
    }
    print("Combined Data = " + str(combined_data))
    json_output = json.dumps(combined_data, indent=4)
    if validate_json(json_output):
        print(f"JSON Validation: Success\n\n{json_output}")
        return json_output
    else:
        return None


def validate_json(json_data):
    """
    Validates JSON data to ensure it is correctly formatted.

    Args:
        json_data (str): JSON data as a string.

    Returns:
        bool: True if the JSON is valid, False otherwise.
    """
    try:
        json.loads(json_data)  # Attempt to parse the JSON
        print("JSON Validation: Success")
        return True
    except json.JSONDecodeError as e:
        print(f"JSON Validation Error: {str(e)}")
        return False

