import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class ActionConverter:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.actions = self._load_actions()
        self.task_params = self._load_task_params()

    def _load_actions(self) -> List[Dict]:
        with open(self.log_file, 'r') as f:
            return json.load(f)

    def _load_task_params(self) -> Dict:
        from tasks.tennis_task import get_prompt
        return get_prompt()[1]

    def override_task_params(self, task_params: Dict):
        self.task_params = task_params

    def generate_code(self) -> str:
        code = f"""import sys
import asyncio
sys.path.insert(0, '.')
from web_voyager.playwright_actions import click, type_text, scroll_until_visible, to_url, setup_browser, wait_for_xpath

async def replay_actions(page, task_params):
"""
        for action in self.actions:
            action_type = action['action']
            params = action['params']
            # Add wait_for_xpath before each action
            if 'xpath' in params:
                code += f"    await wait_for_xpath(page, '{params['xpath']}')\n"
                        
            arg = None
            if 'task_param_used' in params and params['task_param_used']:
                task_param_used = params['task_param_used']
                arg = f"task_params['{task_param_used}']"            
                
            if action_type == 'to_url':
                if not arg:
                    arg = f"'{params['url']}'"
                code += f"    await to_url(page, {arg})\n"
            elif action_type == 'type_text':
                if not arg:
                    arg = f"'{params['text']}'"
                code += f"    await type_text(page, {params['x']}, {params['y']}, {arg})\n"
            elif action_type == 'click':
                if not arg:
                    arg = f"'{params.get('text', '')}'"
                code += f"    await click(page, {params['x']}, {params['y']}, {arg})\n"
            elif action_type == 'scroll':
                if not arg:
                    arg = f"'{params.get('text', '')}'"
                code += f"    await scroll_until_visible(page, {params['x']}, {params['y']}, '{params['direction']}', {params['scroll_direction']}, {arg})\n"
        
        # Add the main function and script execution
        code += f"""
async def main():
    task_params = {self.task_params}
    page = await setup_browser()
    await replay_actions(page, task_params)
    print("Replay completed. Press Enter to close the browser...")
    await asyncio.get_event_loop().run_in_executor(None, input)
    await page.close()

if __name__ == "__main__":
    asyncio.run(main())
"""
        return code

    def save_code(self, output_file: str):
        with open(output_file, 'w') as f:
            f.write(self.generate_code())

import os
username = os.getenv('GTC_USERNAME')
password = os.getenv('GTC_PASSWORD')
    
# Usage example:
converter = ActionConverter('web_voyager/data/actions_log.json')
converter.override_task_params({
        "url": "https://gtc.clubautomation.com/",
        "username": username,
        "password": password,
        "date": "08/23/2024",
        "from_time": "10:00 AM",
        "to_time": "12:00 PM",
        "duration": "30 Min"
    })
converter.save_code('web_voyager/data/generated_replay_actions.py')