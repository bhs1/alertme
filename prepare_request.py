from codegen import CodeGenerator, combined_code
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os


def get_request_replace_func(test_cases):
    os.environ["LANGCHAIN_PROJECT"] = "request_replace_func"
    # Define the prompt for the LLM
    prompt = f"""Write a Python function named that takes an HTTP request string and a
    dictionary where keys are field names and values are the new values for these fields. The function should
    return the updated HTTP request string. Here is an example test input {test_cases[0]['inputs']}.
    
    The request_params_map will be the first element of the dictionary, it will be used to decide which fields to
    replace in the http request. The request_data will be the second item of the dictionary, the output should be
    the updated request_data without the request_params_map.
        
    Note: the field names in request_params_map may not directly correspond
    to fields names in the HTTP request so it's up to you to figure out how to parse and replace the correct fields.
    You may need to use regex to find the fields you are looking for. You may also need to translate the inputs
    into the correct format for the fields.

    """

    code_solution = CodeGenerator().generate_code(prompt, test_cases)
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
def replace_fields(request, request_params_map, func):
    args = {"global_input": str(request_params_map) + "\n\n" + request,
            "global_output": "",
            "debug_output": ""
            }
    exec(func, args)
    return args["global_output"]

