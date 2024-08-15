import os
from dotenv import load_dotenv

load_dotenv()

def get_prompt():
    username = os.getenv('GTC_USERNAME')
    password = os.getenv('GTC_PASSWORD')
    task = f"""Go to url: https://gtc.clubautomation.com/. Use username: {username}, password: {password}.
Task: Search for courts that satisfy the criteria (do not reserve):
 - date: 08/12/2024
 - from_time: 10am
 - to_time: 9pm
 - duration: 60 mins.
 
 Print the available times.
"""
    return task