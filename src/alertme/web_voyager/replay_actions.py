import ast
import asyncio
import json
from pathlib import Path

from alertme.utils.path_utils import get_data_path
from alertme.web_voyager.playwright_actions import setup_browser
from alertme.web_voyager.screenshot_utils import mask_sensitive_data


class ActionReplayer:
    def __init__(self, file_path: Path, page=None, actions_string=""):
        self.page = page
        self.file_path = file_path
        self.actions_string = actions_string

    async def initialize(self):
        if not self.page:
            self.page = await setup_browser()

    async def get_page(self):
        return self.page

    async def close(self):
        await self.page.context.browser.close()

    async def load_actions(self):
        print(f"Loading actions_string: {self.actions_string}")
        if self.file_path:
            with open(self.file_path, "r") as f:
                return json.load(f)
        else:
            # Use ast.literal_eval to safely evaluate the Python literal
            python_obj = ast.literal_eval(self.actions_string)
            # Convert the Python object to proper JSON
            return json.loads(json.dumps(python_obj))

    async def wait_for_xpath(self, xpath):
        await wait_for_xpath(self.page, xpath)

    async def print_action(self, action, params):
        params_copy = params.copy()
        params_copy.pop('parent_html', None)
        print_str = f"Replaying action: {action}, {params_copy}"
        masked_print_str = await mask_sensitive_data(print_str)
        print(masked_print_str)

    async def display_dot(self, x, y):
        await display_red_dot(self.page, x, y)

    async def execute_action(self, action_name, params):
        if action_name == "click":
            await click(self.page, params["x"], params["y"], params["text"], exact=True)
        elif action_name == "type_text":
            await type_text(self.page, params["x"], params["y"], params["text"])
        elif action_name == "scroll":
            await scroll_until_visible(
                self.page,
                params["x"],
                params["y"],
                params["direction"],
                params["scroll_direction"],
                params["text"],
            )
        elif action_name == "go_back":
            await go_back(self.page)
        elif action_name == "to_google":
            await to_google(self.page)
        elif action_name == "to_url":
            await to_url(self.page, params["url"])

    async def replay_actions(self, pause_after_each_action=False):
        await self.initialize()
        try:
            actions_log = await self.load_actions()
            for action in actions_log:
                if pause_after_each_action:
                    input("Press Enter to continue...")
                action_name = action["action"]
                params = action["params"]

                if "xpath" in params:
                    await self.wait_for_xpath(params["xpath"])

                await self.print_action(action_name, params)

                if "x" in params and "y" in params:
                    await self.display_dot(params["x"], params["y"])

                await self.execute_action(action_name, params)
        except Exception as e:
            print(f"An error occurred during replay: {e}")
            await self.close()
            raise


async def main():
    file_path = get_data_path() / 'actions_log.json'
    replayer = ActionReplayer(file_path=file_path)
    await replayer.replay_actions(pause_after_each_action=False)

    print("Replay completed. Press Enter to close the browser...")
    await asyncio.get_event_loop().run_in_executor(None, input)

    # Close the browser
    await replayer.close()


if __name__ == "__main__":
    asyncio.run(main())
