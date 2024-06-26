from bs4 import BeautifulSoup
def func(html_content):
    global debug_output
    debug_output = ""
    
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    debug_output += "Parsed HTML content.\n"
    
    # Initialize the result dictionary
    result = {}
    
    # Find all the time slots
    time_slots = soup.select('#times-to-reserve a')
    debug_output += f"Found {len(time_slots)} time slots.\n"
    
    for slot in time_slots:
        # Find the corresponding activity
        activity = slot.find_previous('b').text.strip().replace(' -', '')
        time = slot.text.strip()
        
        debug_output += f"Activity: {activity}, Time: {time}\n"
        
        # Initialize the activity list if not already present
        if activity not in result:
            result[activity] = []
        
        # Append the time if it is not already in the list
        if time not in result[activity]:
            result[activity].append(time)
        else:
            debug_output += f"Duplicate time found for {activity} at {time}.\n"
    
    debug_output += f"Final result: {result}\n"
    return result

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
