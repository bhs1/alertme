import alertme.web_voyager.tasks.tennis_task as tennis_task
from alertme.code_generators.html_parser_generator import generate_html_parser, save_html_parser
from alertme.web_voyager.code_converter import ActionConverter
from alertme.web_voyager.main import run_web_voyager

if __name__ == "__main__":
    # ================================ Run web voyager to generate action log for tennis task ================================
    run_web_voyager(tennis_task.get_prompt(), tennis_task.get_task_params())
    # ================================ Generate python code from action_log.json ================================
    converter = ActionConverter('web_voyager/data/actions_log.json') 
    converter.save_code('web_voyager/data/generated_replay_actions.py')
    # ================================ Generate html parser code =====================================
    html_parser_code = generate_html_parser()
    save_html_parser(html_parser_code)