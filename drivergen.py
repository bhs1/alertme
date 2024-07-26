import threading
import logging
from operator import itemgetter
import uuid
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph.message import AnyMessage, add_messages
from typing import Dict, TypedDict, List
from typing import Annotated
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import traceback
from dotenv import load_dotenv
import json
from difflib import ndiff
from web_utils import get_html_content, get_xpaths
from playwright.sync_api import sync_playwright
import multiprocessing
import queue
from prompts import get_reflection_prompt, get_reasoning_prompt, get_generate_next_nav_step_prompt, get_generate_prompt, get_code_prompt_tuple
from langgraph_utils import generate_graph_image, print_event
from utils import configure_logging

# Configure logging
configure_logging()

# Constants

CODE_SUFFIX = """
def get_input():
    global global_input
    return global_input

def get_page():
    global page
    return page

def set_output(output):
    global global_output
    global_output = output

if __name__ == "__main__":
    input = get_input()
    page = get_page()
    result = main(input, page)
    set_output(str(result))
"""


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

# Data model


class Code(BaseModel):
    """Code output"""
    correction_summary: str = Field(
        description="Summary of what was corrected to produce this code. Include the ### Improvement Plan from the reflection.")
    code_summary: str = Field(description="Step by step summary of code, in pseudocode form.")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class TestCase(TypedDict):
    inputs: str
    outputs: str


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """

    error: str  # Flag for control flow to indicate whether test error was tripped
    last_test_results: str
    # Each element is [reflection_summary, correction_summary]
    previous_corrections: list[list[str]]
    code_solution: str
    iterations: int
    test_cases: list[TestCase]
    reflection_prompt: str
    reflection_summary: str
    nav_output: dict
    reasoning_output: str
    nav_reasoning_prompt: str
    previous_code_generated: str
    condensed_e2e_nav_code: str
    condensed_e2e_nav_code_summary: str

# Utilities


def combined_code(code_solution: Code) -> str:
    """
    Combines import statements and code into a single executable string.

    Args:
        imports (str): The import statements.
        code (str): The main code block.

    Returns:
        str: The combined code.
    """
    return f"{code_solution.imports}\n{code_solution.code}" + CODE_SUFFIX


class CodeGenerator:
    def __init__(self, user_prompt, use_cached_navigation_code=False, start_at_condense=False):
        ###### Environment Variables ######
        load_dotenv()
        ###### End Environment Variables ######

        # Initialize logging and get run_id
        self.run_id = configure_logging()

        # Select LLM
        self.system_prompt_tuple = (
            "user", f"""Overall task to solve: {user_prompt}
            """)

        self.code_prompt_tuple = get_code_prompt_tuple(CODE_SUFFIX)

        self.code_gen_prompt_template = ChatPromptTemplate.from_messages(
            [
                self.system_prompt_tuple,
                self.code_prompt_tuple,
                ("placeholder", "{messages}"),
            ]
        )

        self.llm = self.code_gen_prompt_template | ChatOpenAI(
            model="gpt-4o", temperature=0)

        # Set to false to make more readable and save resource
        code_llm = ChatOpenAI(
            model="gpt-4o", temperature=0)
        code_llm = code_llm.with_structured_output(
            Code, include_raw=False)
        self.code_gen_chain = self.code_gen_prompt_template | code_llm
        self.builder = StateGraph(GraphState)
        self.use_cached_navigation_code = use_cached_navigation_code
        self.start_at_condense = start_at_condense
        self._setup_graph()

        # Initialize PlaywrightRunner
        self.queue_runner = PlaywrightRunner()

    def close(self):
        self.queue_runner.stop()

    def _setup_graph(self):
        # Define the nodes
        # code_solution solution
        self.builder.add_node("generate", self._generate)
        self.builder.add_node("check_code", self._code_check)  # check code
        self.builder.add_node("reflect", self._reflect)

        # Build graph
        if self.use_cached_navigation_code:
            self.builder.add_node("load_cached_navigation",
                                  self._load_cached_navigation)
            self.builder.set_entry_point("load_cached_navigation")
            self.builder.add_edge("load_cached_navigation", "generate")
        elif self.start_at_condense:
            self.builder.add_node("load_previous_nav_code",
                                  self._load_previous_nav_code_steps)
            self.builder.set_entry_point("load_previous_nav_code")
            self.builder.add_node("condense_nav_code", self._condense_nav_code)
            self.builder.add_edge(
                "load_previous_nav_code", "condense_nav_code")
            self.builder.add_conditional_edges("condense_nav_code",
                                               self._decide_to_proceed_to_final_code_stage,
                                               {
                                                   "continue_navigation": "condense_nav_code",
                                                   "finish_code": "generate"
                                               })
        else:
            self.builder.set_entry_point("generate_next_nav_step")
            self.builder.add_node("exec_nav_script", self._exec_nav_script)
            self.builder.add_node("reason_navigation",
                                  self._reason_next_nav_step)
            self.builder.add_node("generate_next_nav_step",
                                  self._generate_next_nav_step)
            self.builder.add_node("condense_nav_code", self._condense_nav_code)

            self.builder.add_conditional_edges("reason_navigation",
                                               self._decide_to_proceed_to_condense_code_stage,
                                               {
                                                   "continue_navigation": "generate_next_nav_step",
                                                   "finish_code": "condense_nav_code"
                                               })
            self.builder.add_conditional_edges("condense_nav_code",
                                               self._decide_to_proceed_to_final_code_stage,
                                               {
                                                   "continue_navigation": "generate_next_nav_step",
                                                   "finish_code": "generate"
                                               })
            self.builder.add_edge("generate_next_nav_step", "exec_nav_script")
            self.builder.add_edge("exec_nav_script", "reason_navigation")
        self.builder.add_edge("generate", "check_code")
        self.builder.add_edge("reflect", "generate")
        # TODO: Add option to to go back to navigation if needed.
        self.builder.add_conditional_edges(
            "check_code",
            self._decide_to_finish,
            {
                "end": END,
                "retry": "reflect",
            },
        )

        memory = SqliteSaver.from_conn_string(":memory:")
        self.graph = self.builder.compile(checkpointer=memory)
        self._generate_graph_image()
        # Parameters
        self.max_iterations = 10

    def correction_list_to_string(self, correction_list: List[tuple]):
        return """\n\n================================================\n
    """.join([f"Correction Summary: {correction[0]}\nImports:\n{correction[1]}\nCode:\n{correction[2]}" for correction in correction_list])

    def _reflect(self, state: GraphState):
        """
        Reflect on the test case failures and debug analysis.

        Args:
            state (GraphState): The current graph state

        Returns:
            GraphState: Updated state with reflection summary
        """
        # List of code that was generated to successfully get to the final page.
        previous_corrections = state["previous_corrections"]
        last_test_results = state["last_test_results"]
        condensed_e2e_nav_code = state["condensed_e2e_nav_code"]
        condensed_e2e_nav_code_summary = state["condensed_e2e_nav_code_summary"]
        logging.info(
            "---REFLECTING ON ERRORS, DEBUG STATEMENTS, and FORMULATING IMPROVEMENT PLAN---")

        # Ensure only the last three corrections are used, or fewer if less than three exist
        last_three_corrections = previous_corrections[-3:] if len(
            previous_corrections) >= 3 else previous_corrections
        context = f"""\
    Previous Corrections, Reflections and Code diffs:\n\n {self.correction_list_to_string(last_three_corrections)}
        """
        prompt = get_reflection_prompt(
            last_test_results, condensed_e2e_nav_code, condensed_e2e_nav_code_summary)
        messages = [("assistant", context), ("user", prompt)]
        logging.info("Reflecting on code generated using LLM...")
        reflection_summary = self.llm.invoke({"messages": messages}).content
        logging.info(f"Reflection summary: {reflection_summary}")
        return {"reflection_prompt": prompt, "reflection_summary": reflection_summary, "error": "yes"}
# Nodes

    def check_code(self, actual_output, expected_output):
        if actual_output == expected_output:
            return True
        else:
            return False

    def _reason_next_nav_step(self, state: GraphState):
        """
        Take one navigation step towards our final objective
        """
        logging.info("---_reason_next_nav_step---")
        nav_output = state.get("nav_output", {})
        html = nav_output.get("html", "")
        xpaths = nav_output.get("xpaths", "")
        url = nav_output.get("url", "")

        logging.info(f"XPaths: {xpaths}")
        logging.info(f"URL: {url}")

        code_so_far = state.get("previous_code_generated", "")        

        prev_output = nav_output.get("global_output", "No prev output")
        prev_debug = nav_output.get("debug_output", "No prev debug")

        iterations = state["iterations"]
        messages = []
        prompt = get_reasoning_prompt(
            html, xpaths, url, prev_output, prev_debug, code_so_far)
        messages += [
            ("user", prompt)]

        logging.info("Reasoning about next navigation step using LLM...")
        reasoning_output = self.llm.invoke({"messages": messages}).content
        logging.info("Reasoning output: " + reasoning_output)

        iterations = iterations + 1
        return {"reasoning_output": reasoning_output, "nav_reasoning_prompt": prompt}

    def _generate_next_nav_step(self, state: GraphState,):
        """
        Generate code updated with the next navigation step

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, code_solution
        """

        logging.info("---_generate_next_nav_step---")

        iterations = state["iterations"]
        messages = []

        prompt = get_generate_next_nav_step_prompt(
            state.get("reasoning_output", "nothing yet"))
        messages += [
            ("user", prompt)]

        logging.info("Generating next navigation step using prompt:")
        code = self.code_gen_chain.invoke({"messages": messages})
        logging.info("Generated code:")
        logging.info(code.correction_summary)
        logging.info(code.imports)
        logging.info(code.code)

        # Log generated code to file
        with open("data/generated_code_log.txt", "a") as log_file:
            log_file.write(f"Iteration {iterations}:\n")
            log_file.write(f"Correction Summary: {code.correction_summary}\n")
            log_file.write(f"Imports:\n{code.imports}\n")
            log_file.write(f"Code:\n{code.code}\n\n")

        previous_code_generated = state.get("previous_code_generated", "")
        if previous_code_generated is None:
            previous_code_generated = ""
        previous_code_generated += f"""
Iteration {iterations}:
{combined_code(code)}
        """

        iterations = iterations + 1
        return {
            "code_solution": code,
            "iterations": iterations,
            "previous_code_generated": previous_code_generated
        }

    def _exec_nav_script(self, state: GraphState):
        """
        Execute the navigation script using the queue-based runner
        """
        logging.info("---_exec_nav_script---")
        code_solution = state["code_solution"]
        runnable_code = combined_code(code_solution)
        input_string = state["test_cases"][0]["inputs"]
        globals_dict = {'global_input': input_string, 'global_output': None,
                        'debug_output': None, "__name__": "__main__"}
        logging.info(f"Running code:\n {runnable_code}")
        result = self.queue_runner.execute_code(runnable_code, globals_dict)

        nav_output = {
            "global_output": result["global_output"],
            "debug_output": result["debug_output"],
            "html": result["html"],
            "xpaths": result["xpaths"],
            "url": result["url"],
        }
        return {"nav_output": nav_output}

    def _generate(self, state: GraphState,):
        """
        Generate a code solution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, code_solution
        """

        logging.info("---GENERATING CODE SOLUTION---")

        iterations = state["iterations"]
        reflection_prompt = state["reflection_prompt"]
        reflection_summary = state["reflection_summary"]
        condensed_e2e_nav_code = state["condensed_e2e_nav_code"]
        condensed_e2e_nav_code_summary = state["condensed_e2e_nav_code_summary"]
        messages = []
        prompt = get_generate_prompt(
            reflection_prompt, reflection_summary, condensed_e2e_nav_code, condensed_e2e_nav_code_summary)
        messages += [
            ("user", prompt)]
        logging.info("Generating code solution using LLM:")
        code = self.code_gen_chain.invoke({"messages": messages})
        logging.info(f"Generated code:\n {combined_code(code)}")

        previous_corrections = state["previous_corrections"]
        previous_corrections.append(
            [code.correction_summary, code.imports, code.code])

        # Log generated code to file
        with open("data/generated_code_log.txt", "a") as log_file:
            log_file.write(f"Iteration {iterations}:\n")
            log_file.write(f"Correction Summary: {code.correction_summary}\n")
            log_file.write(f"Imports:\n{code.imports}\n")
            log_file.write(f"Code:\n{code.code}\n\n")

        iterations = iterations + 1
        return {"previous_correction": previous_corrections, "code_solution": code, "iterations": iterations}

    def _code_check(self, state: GraphState):
        """
        Check code using a new PlaywrightRunner instance for each test case
        """
        logging.info("---CHECKING CODE---")
        code_solution = state["code_solution"]
        test_cases = state["test_cases"]
        iterations = state["iterations"]
        runnable_code = combined_code(code_solution)

        test_results = []
        num_test_cases = len(test_cases)
        succeeded = 0
        for test_case in test_cases:
            input_data = test_case["inputs"]
            globals_dict = {'global_input': input_data, 'global_output': None,
                            'debug_output': None, "__name__": "__main__"}

            # Create a new PlaywrightRunner instance for each test case
            single_run_runner = PlaywrightRunner()
            result = single_run_runner.execute_single_command(
                runnable_code, globals_dict)
            logging.info(f"Xpaths: {result['xpaths']}")
            logging.info(f"Truncated HTML: {result['html'][:100]}")

            output_match = self.check_code(
                result["global_output"], test_case["outputs"])
            if output_match:
                test_results.append(f"""Correct test case! Your output matched the expected output: {test_case['outputs']}. Successful debug output in case it's useful: {result['debug_output']}
                XPaths: {result['xpaths']}""")
                succeeded += 1
            else:
                test_results.append(f"""Failed test case.
                Actual test case output:
                {result['global_output'] if result['global_output'] else "EMPTY"}
                Expected test case output:
                {test_case['outputs']}
                Debug test case output:
                {result['debug_output']}
                XPaths: {result['xpaths']}
                HTML: {result['html']}""")

        pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
        logging.info(f"Pass rate: {pass_rate}")

        if succeeded == num_test_cases:
            logging.info(
                f"=========== CODE BLOCK CHECK: SUCCEEDED in {iterations} iterations ===========")
            return {"error": "no"}

        logging.info("---CODE BLOCK CHECK: FAILED---")
        responses = "\n".join(
            [f"<test case {i} begin >\n{r}\n</test case>" for i, r in enumerate(test_results)])
        test_result_message = f"""Incorrect submission.\nPass rate: {pass_rate}\nResults:\n{responses}"""
        return {
            "last_test_results": test_result_message,
            "error": "yes",
        }

    def _condense_nav_code(self, state: GraphState):
        """
        Condense the navigation code to a single file

        Generate condensed code based on previous results (and reflection if present).
        If the resulting HTML shows we are on the last page (see previous prompt) then put finish_code in result
        Otherwise, summarize the previous nav steps, come up with a plan to generate the condensed end to end version
        reflect on any errors, and put continue_navigation in result.
        Pass result to output and then the decision node will decide whetehr to try again or finish.
        """
        logging.info("---CONDENSING NAVIGATION CODE---")

        previous_code = state.get("previous_code_generated", "")        
        reasoning_output = state.get("reasoning_output", "")

        # TODO update prompt
        messages = []

        prompt = f"""
        Given the previous code iterations which ran in sequence resulted in us getting to the final page:
        {previous_code}
        Condense these iterations in to a single script (still modular in nature).
        
        Also, if we have failed to condense the functions already, consider this reflection and improverment plan:
        {reasoning_output}
        
        Also, be sure to add the code summary which details each step that the condensed code takes to reach the final page.
        This will be used later to make sure we do not miss a step.
        """
        messages += [
            ("user", prompt)]

        logging.info("Generating condensed code using LLM:")
        code = self.code_gen_chain.invoke({"messages": messages})
        logging.info("Generated code:")
        logging.info(combined_code(code))
        logging.info(f"Code summary: {code.code_summary}")

        logging.info("---_exec_condensed_code---")
        runnable_code = combined_code(code)
        input_string = state["test_cases"][0]["inputs"]
        globals_dict = {'global_input': input_string, 'global_output': None,
                        'debug_output': None, "__name__": "__main__"}
        single_run_runner = PlaywrightRunner()
        result = single_run_runner.execute_single_command(
            runnable_code, globals_dict)

        logging.info("---_reason_condensed_code---")
        html = result.get("html", "")
        xpaths = result.get("xpaths", "")
        url = result.get("url", "")
        prev_output = result.get("global_output", "No prev output")
        prev_debug = result.get("debug_output", "No prev debug")

        logging.info(f"XPaths: {xpaths}")
        logging.info(f"URL: {url}")

        messages = []
        prompt = get_reasoning_prompt(
            html, xpaths, url, prev_output, prev_debug, previous_code)
        messages += [
            ("user", prompt)]

        logging.info("Reasoning about condensed code using LLM...")
        reasoning_output = self.llm.invoke({"messages": messages}).content
        logging.info("Condensed reasoning output:\n " + reasoning_output)

        return {
            "reasoning_output": reasoning_output,
            "condensed_e2e_nav_code": code,
        }

    def _print_event(self, event: dict, _printed: set, max_length=1500):
        print_event(event, _printed, max_length)


# Conditional edges

    def _decide_to_finish(self, state: GraphState):
        """
        Determines whether to finish.

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """
        error = state["error"]
        iterations = state["iterations"]

        if error == "no" or iterations == self.max_iterations:
            logging.info("---DECISION: FINISH---")
            return "end"
        else:
            logging.info("---DECISION: RE-TRY SOLUTION---")
            return "retry"

    def _decide_to_proceed_to_condense_code_stage(self, state: GraphState):
        """
        Decide if we should proceed to the condense stage of code iteration
        """
        reasoning_output = state["reasoning_output"]

        if "finish_code" in reasoning_output:
            navigation_code = state.get("previous_code_generated", "")
            logging.info(f"previous_code_generated: {navigation_code}")
            with open("data/previous_nav_code_steps_generated.txt", "w") as file:
                file.write(navigation_code)
            return "finish_code"
        else:
            return "continue_navigation"

    def _decide_to_proceed_to_final_code_stage(self, state: GraphState):
        """
        Decide if we should proceed to the final stage of code iteration
        """
        reasoning_output = state["reasoning_output"]

        if "finish_code" in reasoning_output:
            condensed_e2e_nav_code = state.get("condensed_e2e_nav_code", None)
            if condensed_e2e_nav_code is None:
                logging.error(
                    "ERROR: No condensed_e2e_nav_code found. Exiting...")
                return "continue_navigation"
            executable_code = combined_code(condensed_e2e_nav_code)
            condensed_e2e_nav_code_summary = condensed_e2e_nav_code.code_summary

            logging.info(
                f"condensed_e2e_nav_code logged to file: {executable_code}")
            logging.info(f"condensed_e2e_nav_code_summary: {condensed_e2e_nav_code_summary}")
            with open("data/previous_e2e_nav_code_generated.txt", "w") as file:
                file.write(executable_code)
            with open("data/condensed_e2e_nav_code_summary.txt", "w") as file:
                file.write(condensed_e2e_nav_code_summary)
            return "finish_code"
        else:
            return "continue_navigation"

    def _generate_graph_image(self):
        generate_graph_image(self.graph)

    def _run_in_subprocess(self, code, input_data):
        globals = {'global_input': input_data, 'global_output': None,
                   'debug_output': None, "__name__": "__main__"}
        exec(code, globals)
        globals['main']()

        result = {
            "global_output": globals["global_output"],
            "debug_output": globals["debug_output"]
        }

        return result

    def _load_cached_e2e_navigation(self, state: GraphState):
        logging.info("Loading cached e2e navigation code...")
        with open("data/previous_e2e_nav_code_generated.txt", "r") as file:
            cached_e2e_navigation_code = file.read()
        with open("data/condensed_e2e_nav_code_summary.txt", "r") as file:
            condensed_e2e_nav_code_summary = file.read()
        return {"condensed_e2e_nav_code": cached_e2e_navigation_code, "condensed_e2e_nav_code_summary": condensed_e2e_nav_code_summary}

    def _load_previous_nav_code_steps(self, state: GraphState):
        logging.info("Loading previous navigation code...")
        with open("data/previous_nav_code_steps_generated.txt", "r") as file:
            previous_nav_code = file.read()
        
        return {"previous_code_generated": previous_nav_code}

    def generate_code(self, test_cases):
        _printed = set()
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": thread_id,
            },
            "recursion_limit": 50
        }
        logging.info("Starting graph...")
        result = self.graph.invoke(
            {"previous_corrections": [], "iterations": 0, "test_cases": test_cases},
            config
        )

        self._print_event(result, _printed)

        return result.get('code_solution')


if __name__ == "__main__":
    from tennis_task import get_activity_times_prompt
    load_dotenv()
    username = 'bensc77'
    password = os.getenv('GGP_PASSWORD')
    input = {"date": "07/23/2024", "interval": "30", "timeFrom": "9am",
             "timeTo": "11pm", "username": username, "password": password}
    expected_output = {
        "Available court times": {
            "Pickleball / Mini Tennis": ["9:30pm"],
            "Tennis": ["7:00pm", "7:30pm", "8:00pm", "9:30pm"]
        }
    }
    test_cases = [{'inputs': input, 'outputs': expected_output}]
    website_string = "https://gtc.clubautomation.com/"
    user_prompt = get_activity_times_prompt(test_cases, website_string)
    code_generator = CodeGenerator(
        user_prompt, start_at_condense=False, use_cached_navigation_code=False)
    code_solution = code_generator.generate_code(test_cases)
    print("RESULT:\n\n" + str(combined_code(code_solution)) + "\n\n")

    result = combined_code(code_solution)

    os.makedirs('data', exist_ok=True)

    with open('data/activity_times_func.py', 'w') as f:
        f.write(result)

    logging.info("Code has been written to data/activity_times_func.py")

    code_generator.close()
