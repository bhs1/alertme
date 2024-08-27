import sys
sys.path.insert(0, '.')
from code_generators.html_parser_generator import read_html_from_file
from web_voyager.tasks.tennis_task import get_username, get_password



from web_voyager.data.generated_replay_actions import run
from data.html_parser_func import func

task_params2 = {
    "url": "https://gtc.clubautomation.com/",
    "username": get_username(),
    "password": get_password(),
    "date": "08/29/2024",
    "from_time": "12pm",
    "to_time": "04:00 PM",
    "duration": "60 Min"
}
run(task_params2)
print(f"\n\nRPA result: {func(read_html_from_file())}\n\n")