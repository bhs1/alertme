import json
import re
def func(input_str):
    global debug_output
    debug_output = ""
    
    try:
        # Attempt to parse the input string as JSON
        input_data = json.loads(input_str.replace("'", '"'))
    except json.JSONDecodeError as e:
        debug_output = str(e)
        return ""
    
    request_params_map = input_data.get('request_params_map', {})
    request_data = input_data.get('request_data', {})
    
    # Update the postData params
    if 'postData' in request_data and 'params' in request_data['postData']:
        for param in request_data['postData']['params']:
            if param['name'] in request_params_map:
                param['value'] = request_params_map[param['name']]
    
    # Update the postData text
    if 'postData' in request_data and 'text' in request_data['postData']:
        post_data_text = request_data['postData']['text']
        for key, value in request_params_map.items():
            post_data_text = re.sub(f"{key}=[^&]*", f"{key}={value}", post_data_text)
        request_data['postData']['text'] = post_data_text
    
    return json.dumps(request_data)

def get_input():
    global global_input
    return global_input

def set_output(output):
    global global_output
    global_output = output

# Main execution
input = get_input()
result = func(input)
set_output(str(result))
