import asyncio
import sys
import traceback
from io import StringIO

import alertme.web_voyager.playwright_actions as playwright_actions
from alertme.web_voyager.screenshot_utils import take_screenshot


class AsyncPlaywrightRunner:
    def __init__(self):
        pass

    async def execute_code_async(self, code, globals_dict):
        self.page = await playwright_actions.setup_browser()
        globals_dict["page"] = self.page
        stdout_capture = StringIO()
        sys.stdout = stdout_capture
        try:
            exec(code, globals_dict)
            await eval("run()", globals_dict)
        except Exception as e:
            globals_dict["debug_output"] = traceback.format_exc()
        finally:
            sys.stdout = sys.__stdout__

        asyncio.sleep(2)  # Sleep for 2 seconds to allow final page to load
        # Append stdout to debug_output
        if "debug_output" not in globals_dict:
            globals_dict["debug_output"] = ""
        globals_dict["debug_output"] += f"\nend of debug_output. stdout:\n{stdout_capture.getvalue()}"

        result = {
            "html": "",
            "screenshot": "",
            "url": self.page.url,
            "global_output": globals_dict.get("global_output"),
            "debug_output": globals_dict.get("debug_output"),
        }

        # Try-catch for content
        try:
            result["html"] = await self.page.content()
        except Exception as e:
            globals_dict["debug_output"] += f"\nError getting page content: {str(e)}"

        # Try-catch for screenshot
        try:
            result["screenshot"] = await take_screenshot(self.page)
        except Exception as e:
            globals_dict["debug_output"] += f"\nError taking screenshot: {str(e)}"

        await self.page.close()
        return result

    def execute_code(self, code, globals_dict):
        result = asyncio.run(self.execute_code_async(code, globals_dict))
        return result


if __name__ == "__main__":
    runner = AsyncPlaywrightRunner()
    result = runner.execute_code(
        """
async def run():
    print("Hello, world!")
    """,
        {},
    )
    print(result)
