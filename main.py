# Step 1: Export HAR file which contains all browsers HTTP requests/responses
# Step 2: Run the script to find the releveant request to be translated to python code.

# TODO(0): Read the paper and see if you can get request fn to work consistently using their ideas.
# TODO(0): But actually for request error the issue could just be the comparison??
# - Or maybe just forcing the LLM to look at a more precise diff between expected and actual.

from bs4 import BeautifulSoup
import httpx  # Add this import
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import re
import utils


# Local libs
import prepare_request
import extract_from_response

DEBUG_CONTEXT_SIZE = 500 
LLM_CONTEXT_WINDOW_SIZE = 2048
debug_mode = os.getenv("DEBUG_MODE", "False").lower() in ('true', '1', 't')

# Global variable to track if the log file has been cleared
log_file_initialized = False

def log(message: str, filename="data/main_trace.txt"):
    global log_file_initialized
    mode = 'a' if log_file_initialized else 'w'
    print(message)
    with open(filename, mode) as file:
        file.write(str(message) + "\n")
    log_file_initialized = True

# =============================== USER INPUT BEGIN ===============================
# Options
USE_CACHED_REQUEST_FUNC = True
USE_CACHED_RESPONSE_FUNC = True
USE_REQUEST_FROM_FILE = False

# USER_INPUT (Temp)
HEADER_FIELDS_TO_REPLACE = {
    "logged_in_phpsessid": '''PHPSESSID=qdn271jpc729o7in4krmhnh2k4; __cf_bm=I9bnj.tJU8BIG3wP4SucQYUhj8DBiO8qwPJKcz3AlRY-1718659823-1.0.1.1-R2E7HSg1U9zS6eG2ukrPKth9nzsxQ9sXXHgycyKDkDPpA2J4FkmCjiq1_4Z00Np8rGiLiQhPH1RPok70VONytw; isLoggedIn=1; SessionExpirationTime=1718688689'''
}

# Read HAR file
har_file_path = 'data/example3.har'

# USER INPUT
REQUEST_PARAMS_MAP = {"date": "06/25/2024",
                      "interval": "60", "timeFrom": "16", "timeTo": "0"}
#REQUEST_PARAMS_MAP = {} # must replace this.
# USER_INPUT
with open("data/request_test1.json", "r") as f:
    request = f.read()
with open("data/request_output.txt", "r") as f:
    expected_output = f.read()
    
# Note output should be a json.dumps string, not a dic. Input is a dict.
REQUEST_TEST_CASES = [(json.loads(request), expected_output),]
print("Expected output: " + str(type(expected_output)))
print("Expected output: " + expected_output)

# USER_INPUT
response_content1 = utils.load_response_data('data/response.txt')
response_content2 = utils.load_response_data('data/response2.txt')
RESPONSE_TEST_CASES = [
    {"inputs": f"{response_content1}",
        "outputs": "{'Pickleball / Mini Tennis': ['9:30pm'], 'Tennis': ['9:30pm']}"},
    {"inputs": f"{response_content2}",
        "outputs": "{'Pickleball / Mini Tennis': ['9:00pm', '9:30pm'], 'Tennis': ['8:00pm', '8:30pm', '9:00pm', '9:30pm']}"},
]


# =============================== USER INPUT END ===============================


############################################################################################
######################### FIND RELEVANT TIMES REQUEST/RESPONSE #############################
############################################################################################


def read_har_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        har_data = json.load(file)
    return har_data

def split_text_into_chunks(text, words_per_chunk):
    words = text.split()
    chunks = [' '.join(words[i:i + words_per_chunk])
              for i in range(0, len(words), words_per_chunk)]
    return chunks


def check_with_llm(text):
    global debug_mode
    # Split the text into chunks of a specified number of words
    chunks = split_text_into_chunks(text, LLM_CONTEXT_WINDOW_SIZE)

    found = False
    response = "no response"
    for chunk in chunks:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                # TODO(P2): There may be cases where there are no available times and then we must make this prompt more interesting. But for now the human should be able to find examples.
                {"role": "user", "content": f"Does the following text contain information about availability times? Keep your response to yes or no \n\n'{chunk}'"}
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


# TODO(https://app.itsdart.com/t/T8NoPJ11waRB, P2): We can try a few candidates and get feedback after subsequent agentic steps.
def meets_requirement(text):        
    return check_with_llm(text)


def extract_filtered_request_response_pairs(har_data):
    log("Extracting filtered request response pairs")
    filtered_pairs = []
    if 'log' in har_data and 'entries' in har_data['log']:
        entries = har_data['log']['entries']
        for entry in entries:            
            request = entry['request']
            response = entry['response']
            response_text = response['content'].get('text', '')

            if meets_requirement(response_text):
                filtered_pairs.append((request, response))
    else:
        log("\033[1;31mERROR: No entries found in HAR file.\033[0m")
    return filtered_pairs


def print_request_response_summary(pairs, total_entries):
    global debug_mode
    total_matched = len(pairs)
    total_not_matched = total_entries - total_matched

    log(f"Total responses that matched the requirement: {total_matched}")
    log(
        f"Total responses that did not match the requirement: {total_not_matched}")

    if debug_mode:
        for i, (request, response) in enumerate(pairs):
            log(f"Entry {i+1} Request:")
            log(f"  Method: {request['method']}")
            log(f"  URL: {request['url']}")
            log(f"  HTTP Version: {request['httpVersion']}")
            log(f"  Headers: {len(request['headers'])} headers")
            log(f"  Query Strings: {len(request.get('queryString', []))} items")
            log(f"  Cookies: {len(request.get('cookies', []))} cookies")
            log(f"  Body Size: {request['bodySize']} bytes")

            log(f"Entry {i+1} Response:")
            log(f"  Status: {response['status']} {response['statusText']}")
            log(f"  HTTP Version: {response['httpVersion']}")
            log(f"  Headers: {len(response['headers'])} headers")
            log(
                f"  Content Size: {response['content'].get('size', 'Unknown')} bytes")
            log(f"  MIME Type: {response['content'].get('mimeType', 'Unknown')}")


def write_first_request_to_file(filtered_pairs, file_path='data/request.json'):
    """
    Writes the first request details from filtered request-response pairs to a file.

    Args:
        filtered_pairs (list): List of filtered request-response pairs.
        file_path (str): Path to the file where the request will be written.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        if filtered_pairs:
            first_request = filtered_pairs[0][0]            
            file.write(json.dumps(first_request))
        else:
            file.write("No request available.")

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
    # updated_cookie = re.sub(r'PHPSESSID=[^;]+', f'PHPSESSID={loggged_in_phpsessid}', old_cookie_string)
    # return updated_cookie
    # TODO:  Automate the login flow, first manually then with LLM. Idea could be for AI to output regex to find and replace the PHPSESSID.
    headers['Cookie'] = loggged_in_phpsessid


def replace_date(body, new_date):
    import re
    # The new date should be in the format MM/DD/YY for the URL, which needs to be URL encoded
    new_date_encoded = new_date.replace('/', '%2F')
    # Replace the date in the body string using a regular expression
    updated_body = re.sub(r'date=[^&]*', f'date={new_date_encoded}', body)
    return updated_body


def extract_playing_times_with_llm(html_content):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""Extract and list all available playing times 
             from the following HTML content:


             {html_content}


            Then in addition, given that the html always follows a similar structure,
            write a python function that takes the html as input and returns the list of playing times.
            If there are multiple activity types, use a dictionary instead of a list and associate times
            with the correct activity.
              """}
        ],
        temperature=0,
        model="gpt-4-turbo",
    )
    response = chat_completion.choices[0].message.content
    log("Extracted Playing Times:" + response)
# Note this was created by the LLM call above!


def extract_playing_times(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    times_table = soup.find('table', {'id': 'times-to-reserve'})
    activities = {}

    if times_table:
        for td in times_table.find_all('td'):
            activity_name = td.find('b').text.strip()
            log(activity_name)
            times = [a.text.strip() for a in td.find_all('a')]
            log(times)
            if activity_name in activities:
                activities[activity_name].extend(times)
            else:
                activities[activity_name] = times

    return activities

def prepare_logged_in_fields(first_request, header_fields_to_replace):
    headers = {header['name']: header['value']
        for header in first_request['headers']}
    print(first_request['headers'])
    # TODO: Update this to be generic after taking care of login flow.
    update_to_logged_in_cookie(headers, header_fields_to_replace['logged_in_phpsessid'])
    
    # Update the headers in the first_request with the new headers
    first_request['headers'] = [{'name': k, 'value': v} for k, v in headers.items()]
    return first_request

def log_curl_command(request_dict, label):
    curl_command = request_to_curl(request_dict)
    log(f"Curl command for {label}:\n{curl_command}")

def prepare_new_request(first_request, request_params_map, process_request_func_str, mode = "LLM"):
    log_curl_command(first_request, "old request")
    log(f"old request: {first_request}")
    first_request = json.loads(prepare_request.replace_fields(first_request, request_params_map, process_request_func_str))
    log_curl_command(first_request, "new request")
    log(f"new request: {first_request}")
    return first_request

def send_request(method, url, headers, body=None):
    # List of headers to exclude (typically handled by the HTTP library)
    exclude_headers = {':method', ':path', ':scheme', ':authority', 'host', 'content-length'}

    # Convert list of headers to dictionary if it's not already a dictionary
    if isinstance(headers, list):
        headers_dict = {
            header['name'].lower(): header['value']
            for header in headers
            if header['name'].lower() not in exclude_headers
        }
    else:
        headers_dict = {
            k.lower(): v
            for k, v in headers.items()
            if k.lower() not in exclude_headers
        }
    
    # Decode byte strings if necessary
    headers_dict = {
        k.decode() if isinstance(k, bytes) else k: 
        v.decode() if isinstance(v, bytes) else v
        for k, v in headers_dict.items()
    }
    
    # Use httpx.Client() for making requests
    with httpx.Client(http2=True) as client:
        # Check the method and call the appropriate function from httpx
        if method.upper() == 'POST':
            response = client.post(url, headers=headers_dict, data=body)
        elif method.upper() == 'GET':
            response = client.get(url, headers=headers_dict)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    
    return response

import json

def request_to_curl(request_dict):
    """
    Convert a request dictionary to a curl command string.
    """
    method = request_dict.get('method', 'POST')
    url = request_dict['url']
    headers = request_dict.get('headers', [])
    body = request_dict.get('postData', {}).get('text', '')

    curl_command = f"curl -X {method} '{url}'"

    for header in headers:
        curl_command += f" -H '{header['name']}: {header['value']}'"

    if body:
        curl_command += f" -d '{body}'"

    return curl_command

def write_request_dict_to_file(request_dict, file_path='data/request_text.txt'):
    """
    Writes the request dictionary as JSON to a file.

    Args:
        request_dict (dict): The request dictionary to be written.
        file_path (str): Path to the file where the JSON will be written.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(request_dict, file, indent=2)
    log(f"Request dictionary written as JSON to {file_path}")

def process_request(request):
    try:
        if isinstance(request, str):
            request_dict = json.loads(request)
        elif isinstance(request, dict):
            request_dict = request
        else:
            raise ValueError("Input must be a JSON string or a dictionary")
        
        print("Converted request to dictionary:", request_dict)
        
        # Write request_dict as JSON to file
        write_request_dict_to_file(request_dict)
        
        # Continue processing with request_dict...
    except json.JSONDecodeError as e:
        print("Error parsing request string to dictionary:", e)
        return None

    if 'method' in request_dict:
        method = request_dict['method']
    url = request_dict['url']
    headers = request_dict['headers']
    body = request_dict.get('postData', {}).get('text', '')
    log(f"""Sending request:
        Method: {method}
        URL: {url}
        Headers: {headers}
        Body: {body}
        """)

    response = send_request(method, url, headers, body)
    log(f"Resent Request Status: {response.status_code}")
    return response.text


def print_and_log(har_data, filtered_request_response_pairs):
    total_entries = len(
        har_data['log']['entries']) if 'log' in har_data and 'entries' in har_data['log'] else 0
    print_request_response_summary(
        filtered_request_response_pairs, total_entries)    
    write_first_request_to_file(filtered_request_response_pairs)


def write_function_to_file(func_str, file_path='data/successful_func.txt'):
    if func_str:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(func_str)
            log(f"Written successful request function to {file_path}")
    else:
        log("No successful request function to write.")

def write_response_test_cases_to_file(test_cases, file_path='data/response_test_cases.txt'):
    """
    Writes the response test cases to a text file with pretty print.

    Args:
        test_cases (list): List of response test cases.
        file_path (str): Path to the file where the test cases will be written.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for test_case in test_cases:
            file.write(f"{test_case}\n\n")
    log(f"Response test cases written to {file_path}")

def write_response_to_file(response_text, filename='data/response_text.txt'):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(response_text)
    log(f"Response text written to {filename}")

if __name__ == "__main__":    
    # Write response test cases to a file
    write_response_test_cases_to_file(RESPONSE_TEST_CASES)

    import utils
    import codegen
    # Setup
    load_dotenv()  # This loads the environment variables from the .env file
    # Initialize the OpenAI client with your API key
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Extract the relevant request/response pair using LLM
    har_data = read_har_file(har_file_path)
    filtered_request_response_pairs = extract_filtered_request_response_pairs(
        har_data)

    # Print and log
    print_and_log(har_data, filtered_request_response_pairs)

    response_text = ""
    
    code_generator = codegen.CodeGenerator()
    
    # Replace fields in the request and resend it.
    if filtered_request_response_pairs:
        first_request = filtered_request_response_pairs[0][0]

        # Extract request params map if no defaults provided by user.
        # Then exit and ask user to run again
        if len(REQUEST_PARAMS_MAP) == 0:
            log("Extracting request params since none provided by user: " + str(REQUEST_PARAMS_MAP))
            request_map = prepare_request.get_request_params_map(first_request)
            log("""Terminating program (success).
                  Rerun using the correct subset of this request params map and enter
                  desired values:\n""" + str(request_map))
            exit()
        else:
            # On second run use cached REQUEST_PARAMS_MAP entered by user.
            log("Using request params map provided by user. REQUEST_PARAMS_MAP: " + str(REQUEST_PARAMS_MAP))
            request_map = REQUEST_PARAMS_MAP

        if len(REQUEST_TEST_CASES) == 0:
            log("We need test cases. Using defaults response.")
            with open('data/response.txt', 'r', encoding='utf-8') as file:
                response_text = file.read()
        else:
            log("Getting request replace function...")
            if USE_CACHED_REQUEST_FUNC:
                with open("data/successful_request_replace_func.py", "r") as f:
                    process_request_func_str = f.read()
            else:
                process_request_func_str = prepare_request.get_request_replace_func(code_generator,
                    REQUEST_TEST_CASES, request_map)
                # If not an empty string, write successful request func to file
                write_function_to_file(process_request_func_str, "data/successful_request_replace_func.py")
            
            #In case you want to use default succesful_fun.
            #with open("data/succesful_func.txt", "r") as f:
            #    process_request_func_str = f.read()            
            # TODO: prepare_headers is not general, fix login flow.
            log("replacing fields in header and body...")
            first_request = prepare_logged_in_fields(first_request, HEADER_FIELDS_TO_REPLACE)
            if USE_REQUEST_FROM_FILE:
                log("Using request from file...")
                with open('data/request.json', 'r') as f:
                    first_request = json.loads(f.read())
            else:
                first_request = prepare_new_request(first_request, REQUEST_PARAMS_MAP, process_request_func_str)
            log("Resending updated request...")
            response_text = process_request(first_request)
            write_response_to_file(response_text)
            
    else:
        with open('data/response.txt', 'r', encoding='utf-8') as file:
            response_text = file.read()
        log("No request-response pairs available to resend. Using default response.")

    # Override response for demo (remove this)
    #with open('data/response.txt', 'r', encoding='utf-8') as file:
    #    response_text = file.read()
    # Extract useful information from the response.
    if response_text is not None:
        log("Getting response parsing function...")
        if USE_CACHED_RESPONSE_FUNC:
            with open("data/successful_response_func.py", "r") as f:
                response_func = f.read()
        else:
            response_func = extract_from_response.generate_response_func(code_generator, RESPONSE_TEST_CASES)
            # If not an empty string, write successful response func to file
            write_function_to_file(response_func, "data/successful_response_func.py")
        log("Parsing results from response...")      
        final_result = extract_from_response.extract_info_from_response(response_text, response_func)
        log("=================== AVAILABLE TIMES FROM RESPONSE ======================")
        # TODO: Get this to show nonempty by fixing cookie (tempfix)
        log(final_result)
