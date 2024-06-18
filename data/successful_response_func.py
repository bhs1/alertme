import re
from bs4 import BeautifulSoup
global debug_output
debug_output = ""

def func(html_content):
    global debug_output
    try:
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        debug_output += "Parsed HTML content.\n"
        
        # Find all the time slots
        time_slots = soup.select('#times-to-reserve a')
        debug_output += f"Found {len(time_slots)} time slots.\n"
        
        # Initialize the result dictionary
        result = {}
        
        # Extract times and activities
        for slot in time_slots:
            activity = slot.find_previous('b').text.strip().replace(' -', '')
            time = slot.text.strip()
            debug_output += f"Activity: {activity}, Time: {time}\n"
            if activity not in result:
                result[activity] = []
            if time not in result[activity]:  # Ensure no duplicates
                result[activity].append(time)
                debug_output += f"Added time {time} to activity {activity}.\n"
            else:
                debug_output += f"Duplicate time {time} for activity {activity} ignored.\n"
        
        return result
    except Exception as e:
        debug_output += f"Exception occurred: {str(e)}\n"
        return {}
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
