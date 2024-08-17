import os
from dotenv import load_dotenv

load_dotenv()

def get_prompt():
    username = os.getenv('GTC_USERNAME')
    password = os.getenv('GTC_PASSWORD')
    task = f"""
Task: 
 (1) Search for courts that satisfy the criteria in the task_params below (do not reserve).
 (2) Print the available times.
"""

    task_params = {
        "url": "https://gtc.clubautomation.com/",
        "username": username,
        "password": password,
        "date": "08/22/2024",
        "from_time": "10:00 AM",
        "to_time": "09:00 PM",
        "duration": "60 Min"
    }
    return task, task_params