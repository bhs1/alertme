from bs4 import BeautifulSoup
debug_output = ""

def func(input_str):
    global debug_output
    
    # Parse the input HTML content
    soup = BeautifulSoup(input_str, 'html.parser')
    
    # Initialize the result dictionary
    result = {}
    
    # Find all the time slots
    time_slots = soup.select('table#times-to-reserve a')
    
    for slot in time_slots:
        activity_info = slot.find_previous('b').text.strip()
        time = slot.text.strip()
        
        # Extract the activity name and correctly strip the ' - '
        if activity_info.endswith(' -'):
            activity_name = activity_info[:-2].strip()
        else:
            activity_name = activity_info.strip()
        
        # Debugging statement
        debug_output += f"Activity Info: {activity_info}, Activity Name: {activity_name}, Time: {time}\n"
        
        if activity_name not in result:
            result[activity_name] = []
        
        # Add the time to the list if it's not already present
        if time not in result[activity_name]:
            result[activity_name].append(time)
    
    return result

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
