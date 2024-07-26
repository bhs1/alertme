from drivergen import CodeGenerator, combined_code, CODE_SUFFIX
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os
import json
import yaml

def format_test_case_to_yaml(test_case):
    # If test_case is already a dict, use it directly
    if isinstance(test_case, dict):
        test_case_dict = test_case
    else:
        # If it's a string, remove outer quotes and parse as JSON
        test_case_dict = json.loads(test_case.strip("'\""))
    
    # Convert the dictionary to YAML
    yaml_string = yaml.dump(test_case_dict, default_flow_style=False)
    
    # Remove the top-level dashes that yaml.dump adds
    yaml_string = yaml_string.replace('---\n', '')
    
    return yaml_string

def get_activity_times_prompt(test_cases, website_string):        
    os.environ["LANGCHAIN_PROJECT"] = "get_activity_times_code"
    
    # Format the first test case to YAML
    formatted_test_case = format_test_case_to_yaml(test_cases[0])
    
    # Define the prompt for the LLM
    prompt = f"""Write a python function that 
    
    (1) Logs in to the website {website_string}
    (2) Navigates to page to find activity times
    (3) Searches for activity times
    (4) Parses activity times and returns a dictionary with activity names as keys and available times as values.
    
    It should ideally be modularized into multiple function calls.
    
    It must pass test cases and here is the first test case:
    
{formatted_test_case}
    """
    return prompt