from playwright.sync_api import sync_playwright
from web_utils import get_html_content, get_xpaths
import time
import sys
import json

def write_to_stderr(data):
    while True:
        try:
            json.dump(data, sys.stderr)
            sys.stderr.flush()
            break
        except BlockingIOError:
            time.sleep(0.1)  # Wait a short time before trying again

def run_playwright_code(code, input_data):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()
        global_scope = {
            "global_input": {"page": page, "input": input_data},
            "global_output": "",
            "global_debug": "",
            "__name__": "__main__"
        }
        try:
            write_to_stderr({"status": "Executing code..."})
            exec(code, global_scope)
            write_to_stderr({"status": "Code execution completed."})
            
            # Print global_output to stdout
            print(global_scope['global_output'])
            
            # Write other data as JSON to stderr
            write_to_stderr({
                "global_debug": global_scope["global_debug"],
                "html": get_html_content(page),
                "xpaths": get_xpaths(page)
            })
            
            # Add a delay to keep the browser open
            time.sleep(5)
        except Exception as e:
            write_to_stderr({"error": f"Exception during execution: {str(e)}"})
        finally:
            write_to_stderr({"status": "Closing browser..."})
            browser.close()

if __name__ == "__main__":
    code = sys.argv[1]
    input_data = json.loads(sys.argv[2])
    
    print("Starting Playwright runner...", file=sys.stderr)
    run_playwright_code(code, input_data)
    print("Playwright runner completed.", file=sys.stderr)
