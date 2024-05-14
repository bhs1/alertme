# Step 1: Export HAR file which contains all browsers HTTP requests/responses
# Step 2: Run the script to find the releveant request to be translated to python code.

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

DEBUG_CONTEXT_SIZE = 500

def read_har_file(file_path):
    # TODO: Figure out how to get the HAR file automatically. Currently we just download it
    # manually.
    with open(file_path, 'r', encoding='utf-8') as file:
        har_data = json.load(file)
    return har_data

# TODO: Make thie requirement powered by LLM.
# TODO: We can try a few candidates and get feedback after subsequent agentic steps.
def meets_requirement(text, requirement, strategy='CONTAINS'):
    if strategy == 'CONTAINS':
        return requirement in text
    elif strategy == 'USE_LLM':
        # Assuming the function to call the LLM is defined elsewhere and handles API interaction
        return check_with_llm(text, requirement)

# TODO(0): Fix error in this function.
def check_with_llm(text, requirement):
    chunks = [text[i:i+2048] for i in range(0, len(text), 2048)]
    found = False
    for chunk in chunks:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Does the following text contain information about gameplay availability times? Keep your response to yes or no \n\n'{chunk}'"}
            ],
            model="gpt-3.5-turbo",
        )
        response = chat_completion.choices[0].message.content.lower()
        if debug_mode:
            print(f"response:\n {response}")  # Print the last response or modify as needed
        if "yes" in response:
            found = True
            break
    return found

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
