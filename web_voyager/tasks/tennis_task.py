import sys

sys.path.insert(0, '.')

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os
import json
load_dotenv()

def get_prompt():
    task = f"""
Task: 
    Search for courts that satisfy the criteria in the task_params below (do not reserve).
    Be sure to set ALL task_params from_time, to_time, date, and duration. Print the results.
"""
    return task

def get_username():
    return os.getenv('GTC_USERNAME')

def get_password():
    return os.getenv('GTC_PASSWORD')

def get_task_params():
    username = os.getenv('GTC_USERNAME')
    password = os.getenv('GTC_PASSWORD')
    task_params = {
        "url": "https://gtc.clubautomation.com/",
        "username": username,
        "password": password,
        "date": "09/04/2024",
        "from_time": "10am",
        "to_time": "09:00 PM",
        "duration": "60 Min"
    }
    return task_params

def get_ai_generated_start_code():
    with open('web_voyager/data/generated_replay_actions.py', 'r') as file:
        return file.read()

# from data.successful_response_func import func
# def get_html_parser():
#     return func

def custom_code_task_guideance():
    return f"""
        Do not attempt to extract any data from the page. We will extract the data in the next step.
    """
    
def get_html_parser_code_gen_prompt(test_cases):
    return f"""Write a function to extract and list all available playing times 
from HTTP response content.

Example HTTP response content:
{test_cases[0]['inputs']}

Example output:
{test_cases[0]['outputs']}
"""

def get_html_response_text():
    file_path = '/Users/bensolis-cohen/Projects/alertme/data/response.txt'
    with open(file_path, 'r') as file:
        return file.read()
def get_empty_response_text():
    file_path = '/Users/bensolis-cohen/Projects/alertme/data/empty_response.txt'
    with open(file_path, 'r') as file:
        return file.read()

def get_available_playing_times():
    return {
        'Pickleball / Mini Tennis': ['2:00pm'],
        'Tennis': [
            '10:00am', '10:30am', '11:00am', '11:30am',
            '12:00pm', '12:30pm', '1:00pm', '1:30pm',
            '2:00pm', '2:30pm', '3:00pm', '3:30pm', '4:00pm'
        ]
    }

def get_html_parser_test_cases():
    return [{'inputs': get_html_response_text(), 'outputs': get_available_playing_times()},
            {'inputs': get_empty_response_text(), 'outputs': {}}]