from codegen import CodeGenerator, combined_code
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os


def get_request_replace_func(request_params_map, test_cases):
    # Define the prompt for the LLM
    prompt = f"""Write a Python function named 'replace_fields' that takes an HTTP request string and a
    dictionary where keys are field names and values are the new values for these fields. The function should
    return the updated HTTP request string. The fields to replace are: {', '.join(request_params_map)}. Here
    is the HTTP request: {test_cases[0]['inputs']}"""

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
    load_dotenv()
    os.environ["LANGCHAIN_PROJECT"] = "codegen"
    # Returns a dictionary with the field names that can be optionally customized by the user.
    prompt = f"""Given this HTTP request {request}, create the corresponding field_list and value_list.
    The fields should only be related to booking activity times. Ignore techincal fields like Host, User-Agent, IDs,
    and things that may not be iterpretable by humans. Try to keep the list at most 10 fields and sort in order of relevance.
    The values should be in the same order as the fields in the field_list. """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    
    field_map_chain = llm.with_structured_output(FieldMap, include_raw=False)
    field_map = field_map_chain.invoke(prompt)
    request_params_map = {field: value for field, value in zip(field_map.field_list, field_map.value_list)}
    return request_params_map


# Works for either header or body.
def replace_fields(request, request_params_map, func):
    return exec(func,
                {"global_input":
                 {"request": request, "request_params_map": request_params_map}
                 }
                )


if __name__ == "__main__":
    with open("data/request.txt", "r") as f:
        request = f.read()
    request_params_map = get_request_params_map(request)
    print(request_params_map)
    # Got this from above.
    REQUEST_PARAMS_MAP = {'date': '05/09/2024', 'interval': '30', 'timeFrom': '15', 'timeTo': '0'}
    #TODO(0): With REQUEST_PARAMS_MAP, create a function to replace these fields in the request.
    # - Includes creating TEST_CASES.
    #TODO(0): Call the function to replace the fields
