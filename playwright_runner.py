from playwright.sync_api import sync_playwright
from web_utils import get_html_content, get_xpaths
import time
import sys
import json
import logging, multiprocessing, queue, threading, traceback



class PlaywrightRunner:
    def __init__(self):
        logging.info("Starting PlaywrightRunner.")
        self.command_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._run_in_subprocess)
        self.process.start()
        logging.info("PlaywrightRunner started.")

    def _run_in_subprocess(self):
        # This method runs in a separate process
        def run_playwright():
            with sync_playwright() as p:
                self.browser = p.chromium.launch(headless=False)
                self.page = self.browser.new_page()
                logging.info("Browser and page initialized.")
                while True:
                    try:
                        command = self.command_queue.get(timeout=1)
                        if command == "STOP":
                            logging.info(
                                "Received STOP command. Exiting loop.")
                            break
                        globals_dict = command["globals_dict"]
                        globals_dict["page"] = self.page
                        exec(command["code"], globals_dict)
                        result = {
                            "html": self.page.content(),
                            "xpaths": get_xpaths(self.page),
                            "url": self.page.url,
                            "global_output": globals_dict.get("global_output"),
                            "debug_output": globals_dict.get("debug_output")
                        }
                        self.result_queue.put(result)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logging.error(
                            f"Error during code execution: {e}", exc_info=True)
                        result = {
                            "html": self.page.content(),
                            "xpaths": get_xpaths(self.page),
                            "url": self.page.url,
                            "global_output": f"Error during code execution: {e}\n",
                            "debug_output": f"Error during code execution: {e}\n{traceback.format_exc()}"
                        }
                        self.result_queue.put(result)
                self.browser.close()
                logging.info("Browser closed.")

        # Run Playwright in a separate thread to isolate it from any asyncio loop
        playwright_thread = threading.Thread(target=run_playwright)
        playwright_thread.start()
        playwright_thread.join()

    def execute_code(self, code, globals_dict):
        logging.info(f"Executing code...")
        self.command_queue.put({"code": code, "globals_dict": globals_dict})
        result = self.result_queue.get()
        logging.info(f"Execution result received.")
        return result

    def execute_single_command(self, code, globals_dict):
        logging.info("Executing single command...")
        logging.info(f"Code:\n {code}")
        logging.info(f"Globals dict:\n {globals_dict}")
        self.command_queue.put({"code": code, "globals_dict": globals_dict})
        result = self.result_queue.get()
        logging.info("Execution result received.")
        self.stop()
        return result

    def stop(self):
        logging.info("Stopping PlaywrightRunner.")
        self.command_queue.put("STOP")
        self.process.join()
        logging.info("PlaywrightRunner stopped.")


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
