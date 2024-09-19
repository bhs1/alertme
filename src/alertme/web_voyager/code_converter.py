import os
import json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from alertme.utils.path_utils import get_data_path

load_dotenv()


class ActionConverter:
    def __init__(self, logfile_path: Path, task_params={}):
        self.logfile_path = logfile_path
        self.actions = self._load_actions()
        self.task_params = task_params

    def _load_actions(self) -> List[Dict]:
        with open(self.logfile_path, 'r') as f:
            return json.load(f)

    def override_task_params(self, task_params: Dict):
        self.task_params = task_params

    def generate_code(self) -> str:
        code = f"""import sys
import asyncio
sys.path.insert(0, '.')
from web_voyager.playwright_actions import click, type_text, scroll_until_visible, to_url, setup_browser, wait_for_xpath, save_page_html

async def main(page, task_params):
"""
        for i, action in enumerate(self.actions):
            action_type = action['action']
            params = action['params']
            reflection = action.get('reflection', {})

            # Add reflection information as comments
            code += f"    # Intention: {reflection.get('intention_of_action', 'N/A')}\n"
            code += f"    # Result: {reflection.get('result_of_action', 'N/A')}\n"
            code += f"    # Success/Failure: {reflection.get('success_or_failure_result', 'N/A')}\n"

            # Add wait_for_xpath before each action
            if 'xpath' in params:
                code += f"    await wait_for_xpath(page, '{params['xpath']}')\n"

            arg = ""
            if 'task_param_used' in params and params['task_param_used'] and params['task_param_used'] != 'None' and params['task_param_used'] != "error_task_param_used_not_found":
                task_param_used = params['task_param_used']
                arg = f"task_params['{task_param_used}']"
                code += f"    text_arg = {arg}\n"
                if 'task_param_code' in action and action['task_param_code'] and action['task_param_code'] != "null":
                    indented_code = '\n'.join(
                        '    ' + line for line in action['task_param_code'].split('\n'))
                    code += f"""
{indented_code}
    try:
        text_arg = convert(text_arg)
    except Exception as e:
        print(f"Error converting task parameter: {{e}}")
        text_arg = {arg}
""".replace("convert(", f"convert{i}(")
            else:
                code += f"    text_arg = \"{arg}\"\n"
            if action_type == 'to_url':
                if not arg:
                    arg = f"'{params['url']}'"
                code += f"    await to_url(page, text_arg)\n"
            elif action_type == 'type_text':
                if not arg:
                    arg = f"'{params['text']}'"
                code += f"    await type_text(page, {params['x']}, {params['y']}, text_arg)\n"
            elif action_type == 'click':
                if not arg:
                    arg = f"'{params.get('text', '')}'"
                code += f"    await click(page, {params['x']}, {params['y']}, text_arg)\n"
            elif action_type == 'scroll':
                if not arg:
                    arg = f"'{params.get('text', '')}'"
                code += f"    await scroll_until_visible(page, {params['x']}, {params['y']}, '{params['direction']}', {params['scroll_direction']}, text_arg)\n"

            code += "\n"  # Add a blank line between actions for readability

        # Add the main function and script execution
        code += f"""

async def async_run(task_params):
    page = await setup_browser()
    await main(page, task_params)
    await asyncio.sleep(1)
    await save_page_html(page)
    await page.close()
    
def run(task_params):
    asyncio.run(async_run(task_params))

# For testing only
if __name__ == "__main__":
    run({self.task_params})
    # print("Press Enter to exit...")
    # input()
    
"""
        return code

    def save_code(self, output_file: Path):
        with open(output_file, 'w') as f:
            f.write(self.generate_code())


if __name__ == "__main__":
    username = os.getenv('GTC_USERNAME')
    password = os.getenv('GTC_PASSWORD')

    # Usage example:
    from alertme.web_voyager.tasks.tennis_task import get_task_params
    converter = ActionConverter(get_data_path() / 'actions_log.json')
    converter.override_task_params(get_task_params())
    logfile_path = get_data_path() / 'generated_replay_actions.py'
    converter.save_code(get_data_path() / 'generated_replay_actions.py')
