import json
import urllib.parse
debug_output = ""

def func(input_str):
    global debug_output
    debug_output = ""
    try:
        # Parse the input JSON string
        data = json.loads(input_str)
    except json.JSONDecodeError as e:
        debug_output += f"JSONDecodeError: {str(e)}\n"
        return "SUSPECT_BAD_TEST_CASE, REASON: Invalid JSON input"

    request_params_map = data.get("request_params_map", {})
    request_data = data.get("request_data", {})

    debug_output += f"request_params_map: {request_params_map}\n"
    debug_output += f"request_data before update: {request_data}\n"

    # Update the request_data based on request_params_map
    if "postData" in request_data and "params" in request_data["postData"]:
        for param in request_data["postData"]["params"]:
            if param["name"] in request_params_map:
                param["value"] = request_params_map[param["name"]]

        # Generate the 'text' field in postData
        params_list = []
        for param in request_data["postData"]["params"]:
            encoded_param = f"{urllib.parse.quote(param['name'])}={urllib.parse.quote(param['value'])}"
            params_list.append(encoded_param)
        request_data["postData"]["text"] = "&".join(params_list)

    debug_output += f"request_data after update: {request_data}\n"

    return json.dumps(request_data)

def get_input():
    global global_input
    return global_input

def set_output(output):
    global global_output
    global_output = output

if __name__ == "__main__":
    input = get_input()
    result = func(input)
    set_output(str(result))
