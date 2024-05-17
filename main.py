# Step 1: Export HAR file which contains all browsers HTTP requests/responses
# Step 2: Run the script to find the releveant request to be translated to python code.

import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import re

DEBUG_CONTEXT_SIZE = 500
LLM_CONTEXT_WINDOW_SIZE = 2048

############################################################################################
######################### FIND RELEVANT TIMES REQUEST/RESPONSE #############################
############################################################################################
def read_har_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        har_data = json.load(file)
    return har_data

def split_text_into_chunks(text, words_per_chunk):
    words = text.split()
    chunks = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks

def check_with_llm(text, requirement):
    # Split the text into chunks of a specified number of words
    chunks = split_text_into_chunks(text, LLM_CONTEXT_WINDOW_SIZE)
    
    found = False
    for chunk in chunks:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Does the following text contain information about gameplay availability times? Keep your response to yes or no \n\n'{chunk}'"}
            ],
            temperature=0,
            model="gpt-3.5-turbo",
        )
        response = chat_completion.choices[0].message.content.lower()
        if "yes" in response:
            found = True
            break
    if debug_mode:
        print(f"response:\n {response}")
    return found


# TODO(P2): We can try a few candidates and get feedback after subsequent agentic steps.
def meets_requirement(text, requirement, strategy='CONTAINS'):
    if strategy == 'CONTAINS':
        return requirement in text
    elif strategy == 'USE_LLM':
        # Assuming the function to call the LLM is defined elsewhere and handles API interaction
        return check_with_llm(text, requirement)


def extract_filtered_request_response_pairs(har_data, requirement, strategy='CONTAINS'):
    filtered_pairs = []
    if 'log' in har_data and 'entries' in har_data['log']:
        entries = har_data['log']['entries']
        for entry in entries:
            request = entry['request']
            response = entry['response']
            response_text = response['content'].get('text', '')
            
            if meets_requirement(response_text, requirement, strategy):
                filtered_pairs.append((request, response))
    
    return filtered_pairs

def print_request_response_summary(pairs, total_entries, requirement):
    total_matched = len(pairs)
    total_not_matched = total_entries - total_matched

    print(f"Total responses that matched the requirement: {total_matched}")
    print(f"Total responses that did not match the requirement: {total_not_matched}")

    for i, (request, response) in enumerate(pairs):
        print(f"Entry {i+1} Request:")
        print(f"  Method: {request['method']}")
        print(f"  URL: {request['url']}")
        print(f"  HTTP Version: {request['httpVersion']}")
        print(f"  Headers: {len(request['headers'])} headers")
        print(f"  Query Strings: {len(request.get('queryString', []))} items")
        print(f"  Cookies: {len(request.get('cookies', []))} cookies")
        print(f"  Body Size: {request['bodySize']} bytes")
        
        print(f"Entry {i+1} Response:")
        print(f"  Status: {response['status']} {response['statusText']}")
        print(f"  HTTP Version: {response['httpVersion']}")
        print(f"  Headers: {len(response['headers'])} headers")
        print(f"  Content Size: {response['content'].get('size', 'Unknown')} bytes")
        print(f"  MIME Type: {response['content'].get('mimeType', 'Unknown')}")

        # Extracting and printing context around the requirement
        response_text = response['content'].get('text', '')
        index = response_text.find(requirement)
        if index != -1:
            start = max(0, index - DEBUG_CONTEXT_SIZE)
            end = min(len(response_text), index + DEBUG_CONTEXT_SIZE + len(requirement))
            context = response_text[start:end]
            print(f"  Context around '{requirement}': {context}")
        print("")

# Setup
load_dotenv()  # This loads the environment variables from the .env file
# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Setup
debug_mode = os.getenv("DEBUG_MODE", "False").lower() in ('true', '1', 't')

# Usage
har_file_path = 'data/example2.har'
har_data = read_har_file(har_file_path)
requirement = '0pm'
strategy = 'USE_LLM'  # or 'CONTAINS'
filtered_request_response_pairs = extract_filtered_request_response_pairs(har_data, requirement, strategy)
total_entries = len(har_data['log']['entries']) if 'log' in har_data and 'entries' in har_data['log'] else 0
print_request_response_summary(filtered_request_response_pairs, total_entries, requirement)


############################################################################################
##############RESEND THE REQUEST AS IS AND TEST RESPONSE IS THE SAME #######################
############################################################################################
import requests

# TODO(0): Verify that the issue is login token (by logging in and replacing token in request) then continue the flow.
# TODO(P2): Solve the login flow using LLM
# Resend the first request from the filtered pairs
def update_to_logged_in_cookie(headers, loggged_in_phpsessid):
    """
    Updates the PHPSESSID in a cookie string with a new PHPSESSID value.
    Args:
    old_cookie_string (str): The original cookie string from the headers.
    loggged_in_phpsessid (str): The new PHPSESSID value to be inserted.
    Returns:
    str: Updated cookie string with the new PHPSESSID.
    """
    # Regex to find and replace the PHPSESSID value
    # Could be something like this
    #updated_cookie = re.sub(r'PHPSESSID=[^;]+', f'PHPSESSID={loggged_in_phpsessid}', old_cookie_string)
    #return updated_cookie
    # TODO: fix and automate this. Idea could be for AI to output regex to find and replace the PHPSESSID.
    headers['Cookie'] = '''PHPSESSID=rjes9bojneodauocf5uqdpr09n; __cf_bm=y94bfz7zVOCAB5ZZ\
    Vyr4xyDp88aLX2eTk8VERHywlV8-1715922970-1.0.1.1-kmirNpFXvlwYkcoDYHPIsD0RmHtFSUm71MLjBmZ\
    DciFZyYgpHA8isACN2PWL_yyUIXkdet82CH6AhlhmIAt.zA; isLoggedIn=1; SessionExpirationTime=17\
    15951794'''

if filtered_request_response_pairs:
    first_request = filtered_request_response_pairs[0][0]
    method = first_request['method']
    url = first_request['url']
    headers = {header['name']: header['value'] for header in first_request['headers']}

    update_to_logged_in_cookie(headers, '')
    
    # TODO(0): Replace date with 5/22 and times with 10am-11pm and verify the request works.
    body = first_request.get('postData', {}).get('text', '')

    # Depending on the method, send the request
    if method.lower() == 'get':
        response = requests.get(url, headers=headers)
    elif method.lower() == 'post':
        response = requests.post(url, headers=headers, data=body)
    else:
        response = requests.request(method, url, headers=headers, data=body)

    # Print the status code and response text to verify if it matches the original response
    print(f"Resent Request Status: {response.status_code}")

    # TODO(0): Get LLM generated code to extract times from response and print.
    # Key is not getting times using LLM directly, but instead having LLM generate code to get the times.
    # print(f"Resent Request Response Text: {response.text}...")
else:
    print("No request-response pairs available to resend.")

############################################################################################
############################ SEND THE REQUEST WITH NEW PARAMS ##############################
############################################################################################


# TODO: Implement (in a general LLM powered way)

############################################################################################
################  PARSE THE RESPONSE AND COMPARE TO USER FILTER ############################
############################################################################################

# TODO: Implement (in a general LLM powered way)


# TODO: Try to generalize the script from tennis bot to passport bot then golf bot.
