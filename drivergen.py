import sys
sys.path.insert(0, '.')
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
from web_utils import get_html_content
from playwright.async_api import async_playwright
import multiprocessing
import queue
from prompts import get_reflection_prompt, get_reasoning_prompt, get_generate_next_nav_step_prompt, get_generate_prompt_v2, get_code_prompt_tuple
from langgraph_utils import generate_graph_image, print_event
from utils import configure_logging
from async_playwright_runner import AsyncPlaywrightRunner
from langchain_core.tracers.langchain import wait_for_all_tracers

# Configure logging
configure_logging()

# Constants

CODE_SUFFIX = """
import web_voyager.playwright_actions_core as playwright_actions_core

def get_input():
    global global_input
    return global_input

def get_page():
    global page
    return page

def set_output(output):
    global global_output
    global_output = output

async def run():
    input = get_input()
    page = get_page()
    result = await main(input, page)
    set_output(str(result))
    
"""

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
    screenshot: str
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
    def __init__(self, user_prompt, use_cached_navigation_code=False, start_at_condense=False, html_parser=None):
        ###### Environment Variables ######
        load_dotenv()
        ###### End Environment Variables ######
        
        self.html_parser = html_parser

        # Initialize logging and get run_id
        self.run_id = configure_logging()

        # Select LLM
        self.system_prompt_tuple = (
            "user", f"""Overall task to solve: {user_prompt}
            """)

        self.code_prompt_tuple = get_code_prompt_tuple(CODE_SUFFIX)
        
        self.image_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "Screenshot of test1 screen after code execution:"),
                ("user", [
                    {
                        "type" : "image_url",
                        "image_url" : {"url": "data:image/png;base64,{img}"}
                    },
                    
                ])
            ],
        )

        self.base_llm = ChatOpenAI(
            model="gpt-4o", temperature=0)

        # Set to false to make more readable and save resource
        self.code_llm = self.base_llm.with_structured_output(
            Code, include_raw=False)
        self.builder = StateGraph(GraphState)
        self.use_cached_navigation_code = use_cached_navigation_code
        self.start_at_condense = start_at_condense
        self._setup_graph()

        # Initialize PlaywrightRunner
        self.queue_runner = AsyncPlaywrightRunner()

    def create_prompt_template(self, messages):
        return ChatPromptTemplate.from_messages(
            messages + [("placeholder", "{{tmp}}")],
            template_format = "mustache"
        )
    def get_code_gen_prompt_template(self, message):
        return ChatPromptTemplate.from_messages(
            [
                self.system_prompt_tuple,
                self.code_prompt_tuple,
                ("user", message), # Abandon all the invoke params and templates and go back to python function with formatted string.
            ],
            template_format = "mustache"
        )
    def get_code_gen_prompt_messages(self, message):
        return [self.system_prompt_tuple, self.code_prompt_tuple, ("user", message)]
    def close(self):
        self.queue_runner.stop()

    def _setup_graph(self):
        # Define the nodes
        # code_solution solution
        self.builder.add_node("generate", self._generate)
        self.builder.add_node("check_code", self._code_check)  # check code
        self.builder.add_node("reflect", self._reflect)

        self.builder.set_entry_point("generate")        
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
        logging.info(
            "---REFLECTING ON ERRORS, DEBUG STATEMENTS, and FORMULATING IMPROVEMENT PLAN---")

        # Ensure only the last three corrections are used, or fewer if less than three exist
        last_three_corrections = previous_corrections[-3:] if len(
            previous_corrections) >= 3 else previous_corrections
        context = f"""\
    Previous Corrections, Reflections and Code diffs:\n\n {self.correction_list_to_string(last_three_corrections)}
        """
        prompt = get_reflection_prompt(
            last_test_results)
        messages = context + "\n" + prompt        
        code_prompt = self.get_code_gen_prompt_template(messages)
        logging.info("Reflecting on code generated using LLM...")
        if state.get("screenshot", ""):
            concatenated_prompts =  code_prompt + self.image_prompt
            full_chain = concatenated_prompts | self.base_llm
        else:
            full_chain = self.llm
            logging.info("No screenshot provided, using code generation prompt only.")        
        reflection_summary = full_chain.invoke({
            "img": state.get("screenshot", ""),            
        }).content
        logging.info(f"Reflection summary: {reflection_summary}")
        logging.info("Waiting for all tracers to finish...")
        wait_for_all_tracers()
        logging.info("All tracers finished.")
        return {"reflection_prompt": prompt, "reflection_summary": reflection_summary, "error": "yes"}
# Nodes

    def check_code(self, actual_output, expected_output):
        if actual_output == expected_output:
            return True
        else:
            return False

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
        messages = []
        prompt = get_generate_prompt_v2(reflection_prompt, reflection_summary)        
        code_prompt_messages = self.get_code_gen_prompt_messages("")
        messages = [("user", prompt)] + code_prompt_messages
        logging.info("Generating code solution using LLM:")
        code = self.code_llm.invoke(messages)
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
        screenshot = None
        for test_case in test_cases:
            input_data = test_case["inputs"]
            globals_dict = {'global_input': input_data, 'global_output': None,
                            'debug_output': None, "__name__": "__main__"}

            # Create a new PlaywrightRunner instance for each test case
            single_run_runner = AsyncPlaywrightRunner()
            result = single_run_runner.execute_code(
                runnable_code, globals_dict)
            # TODO: generalize this to more tests.
            screenshot = result['screenshot']            
            logging.info(f"Truncated HTML: {result['html'][:100]}")
            output = result['global_output']
            if self.html_parser is not None:
                parsed_html = self.html_parser(result['html'])
                logging.info(f"Parsed HTML: {parsed_html}")
                output = parsed_html            
            # TODO: add screenshot test result prompt to optimize.

            output_match = self.check_code(
                output, test_case["outputs"])
            if output_match:
                test_results.append(f"""Correct test case!
Your output matched the expected output: {test_case['outputs']}.
Successful debug output in case it's useful: {result['debug_output']}""")
                succeeded += 1
            else:
                test_results.append(f"""Failed test case.
                Actual test case output:
                {output if output else "EMPTY"}
                Expected test case output:
                {test_case['outputs']}
                Debug test case output:
                {result['debug_output']}""")

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
            "screenshot": screenshot
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

    def _generate_graph_image(self):
        generate_graph_image(self.graph)

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