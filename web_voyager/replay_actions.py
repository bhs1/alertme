import json
import asyncio
from playwright_actions import *
from playwright_actions import setup_browser

async def replay_actions(page, actions_log_path):    
    # Read actions_log from file
    with open(actions_log_path, "r") as f:
        actions_log = json.load(f)
    
    for action in actions_log:
        action_name = action["action"]
        params = action["params"]
        if "xpath" in params:
            try:
                print(f"Waiting for xpath: {params['xpath']}")
                locator = page.locator(f"xpath={params['xpath']}")
                # Wait for at least one element to be visible
                start_time = asyncio.get_event_loop().time()
                await locator.first.wait_for(state="visible", timeout=100000) # 100 secs
                end_time = asyncio.get_event_loop().time()
                print(f"Found {await locator.count()} elements matching the XPath")
                print(f"Time waited: {end_time - start_time:.2f} seconds")
            except Exception as e:
                print(f"Warning: XPath '{params['xpath']}' not found within 10 seconds: {e}")

        print(f"Replaying action: {action['action']}, {action['params']}")

        if action_name == "click":
            await click(page, params["x"], params["y"])
        elif action_name == "type_text":
            await type_text(page, params["x"], params["y"], params["text"])
        elif action_name == "scroll":
            await scroll(page, params["x"], params["y"], params["direction"], params["scroll_direction"])
        elif action_name == "go_back":
            await go_back(page)
        elif action_name == "to_google":
            await to_google(page)
        elif action_name == "to_url":
            await to_url(page, params["url"])
        #await asyncio.sleep(1)
            
if __name__ == "__main__":
    async def main():
        page = await setup_browser()
        await replay_actions(page, "/Users/bensolis-cohen/Projects/alertme/web_voyager/data/actions_log.json")
        
        print("Replay completed. Press Enter to close the browser...")
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        # Close the browser
        await page.context.browser.close()
    
    asyncio.run(main())