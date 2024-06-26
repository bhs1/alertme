import json
def func(input_str):
    global debug_output
    debug_output = ""
    try:
        data = json.loads(input_str)
    except json.JSONDecodeError as e:
        debug_output += f"JSONDecodeError: {str(e)}\n"
        return ""
    
    request_params_map = data.get("request_params_map", {})
    request_data = data.get("request_data", {})
    
    debug_output += f"request_params_map: {request_params_map}\n"
    debug_output += f"request_data: {request_data}\n"
    
    # Update the postData params based on request_params_map
    if "postData" in request_data and "params" in request_data["postData"]:
        for param in request_data["postData"]["params"]:
            if param["name"] in request_params_map:
                param["value"] = request_params_map[param["name"]]
        
        # Generate the text field without URL-encoding
        params = request_data["postData"]["params"]
        text = "&".join(f"{param['name']}={param['value']}" for param in params)
        request_data["postData"]["text"] = text
        
        debug_output += f"Updated params: {params}\n"
        debug_output += f"Generated text: {text}\n"
    
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
