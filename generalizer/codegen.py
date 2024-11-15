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
from dotenv import load_dotenv
import json
from difflib import ndiff
import deepdiff


# Constants

CODE_SUFFIX = """

def get_input():
    global global_input
    return global_input

def set_output(output):
    global global_output
    global_output = output

if __name__ == "__main__":
    input = get_input()
    result = func(input)
    set_output(str(result))
"""

CODE_PREFIX = "__name__ = '__main__'\n"

def exec_as_main(code: str, global_scope: dict = {}):
    exec(CODE_PREFIX + code, global_scope)

# Data model
class Code(BaseModel):
    """Code output"""
    correction_summary: str = Field(
        description="Summary of what was corrected to produce this code. Include the ### Improvement Plan from the reflection.")
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
    def __init__(self):
        ###### Environment Variables ######
        load_dotenv()
        ###### End Environment Variables ######

        # Initialize log file flag
        self.log_file_initialized = False

        # Select LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.code_gen_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("user",
                 f"""
    Coding agent general guideance:
         - You are an expert programmer. Ensure any code you provide can be executed with all
         required imports and variables defined. Structure your answer: 1) a prefix describing
         any corrections, 2) the imports, 3) the functioning code block. The imports should only
         contain import statements. E.g. import math, import sys, etc., not actual code.
    Formatting the function you will write:
         - We will append the following code suffix to whatever code you generate:
         {CODE_SUFFIX} so make sure you call your function "func" and expect it to be called as such with a dictionary argument.
         Don't generate the code suffix, we will append it for you after you generate func(dict).
         """
                 ),
                ("placeholder", "{messages}"),
            ]
        )

        # Set to false to make more readable and save resource
        self.code_gen_chain = self.code_gen_prompt_template | self.llm.with_structured_output(
            Code, include_raw=False)
        self.builder = StateGraph(GraphState)
        self._setup_graph()

    def set_code_executor(self, code_executor):
        self.code_executor = code_executor

    def set_code_checker(self, code_checker):
        self.code_checker = code_checker
        
    def set_additional_params(self, additional_params):
        self.additional_params = additional_params
    
    def _setup_graph(self):
        # Define the nodes
        # code_solution solution
        self.builder.add_node("generate", self._generate)
        self.builder.add_node("check_code", self._code_check)  # check code
        self.builder.add_node("reflect", self._reflect)

        # Build graph
        self.builder.set_entry_point("generate")
        self.builder.add_edge("generate", "check_code")
        self.builder.add_edge("reflect", "generate")
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

    # TODO: Add a field called context that is the context we want to keep. This should be distinct from messages
    # Which is only the directive for the next instruction.
    def _reflect(self, state: GraphState):
        """
        Reflect on the test case failures and debug analysis.

        Args:
            state (GraphState): The current graph state

        Returns:
            GraphState: Updated state with reflection summary
        """
        previous_corrections = state["previous_corrections"]
        last_test_results = state["last_test_results"]        

        self.log(
            "---REFLECTING ON ERRORS, DEBUG STATEMENTS, and FORMULATING IMPROVEMENT PLAN---")
        # Ensure only the last three corrections are used, or fewer if less than three exist
        last_three_corrections = previous_corrections[-3:] if len(previous_corrections) >= 3 else previous_corrections
        context = f"""\
    Previous Corrections, Reflections and Code diffs:\n\n {self.correction_list_to_string(last_three_corrections)}
        """
        self.log("Context:\n" + context)
        prompt = f"""(1) Determine what went wrong in each of the following test results: {last_test_results}.
Explain for each failing test case how does the actual output differ from the expected \
output (be very specific here, feel free to take multiple sentences)? And why do you think that is? \
If useful, consider the Debug Output when reasoning about your response.

Note: You are allowed to be skeptical of the test case. If you have high confidence that there is an issue with the test \
case expected output e.g. incorrectly set fields, you should say so. The same is true for input. But please don't stop \
just call it out and continue. In this case you should output SUSPECT_BAD_TEST_CASE, REASON: <reason>.

     
    (2) Construct an improvement plan to fix the test failures. When formulating this plan, observe the previous solutions
    shown above and try not to repeat fixes that we already know do not lead to passing the test cases. Relfect explicitly on what
    we've tried before and whether it made a difference. If it did not make a difference, why not?

    (3) Sketch out the specific pseudocode of the fixes you plan to implement. Feel free to include actual code snipits."""
        messages = [self.system_prompt_tuple,
                    ("assistant", context), ("user", prompt)]
        reflection_summary = self.llm.invoke(messages).content
        return {"reflection_prompt": prompt, "reflection_summary": reflection_summary, "error": "yes"}
# Nodes

    def _generate(self, state: GraphState,):
        """
        Generate a code solution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, code_solution
        """

        self.log("---GENERATING CODE SOLUTION---")
        # TODO(P2): Cap lookback at previous solutions to 3 as shown to work in the AlphaCodium paper.

        # Correction includes reflection, code diff
        iterations = state["iterations"]
        error = state["error"]
        reflection_prompt = state["reflection_prompt"]
        reflection_summary = state["reflection_summary"]
        messages = [self.system_prompt_tuple]
        if (error == "yes"):
            self.log("--------- Rewriting code with print statements -----------")
            prompt = f"""We just gave you this prompt:\n\n {reflection_prompt} \n\n and as a result \
of the above prompt you gave this relfection summary:\n\n {reflection_summary}\n
Main directive: Rewrite the code to fix the test cases. Be sure to follow the plan \
in the reflection summary.
        
IMPORTANT: Also, print out intermediate variable names for debugging purposes. Instead of print() append \
to global variable debug_output. At first debug_output can be simple, but as you fail a test \
more times, you should add more and more details and variables to debug_output until we find \
the issue. Remember to define "global debug_output" at the top of any scope that updates debug_output.
                 """
            self.log("Generating solution using prompt:")
            self.log(prompt)
            messages += [
                ("user", prompt)]

        # Solution        
        code = self.code_gen_chain.invoke({"messages": messages})            
        previous_corrections = state["previous_corrections"]
        previous_corrections.append([code.correction_summary, code.imports, code.code])

        # Increment
        iterations = iterations + 1
        return {"previous_correction": previous_corrections, "code_solution": code, "iterations": iterations}

    def _code_check(self, state: GraphState):
        """
        Check code

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        self.log("---CHECKING CODE---")

        # State
        code_solution = state["code_solution"]
        test_cases = state["test_cases"]
        iterations = state["iterations"]

        # Get solution components
        imports = code_solution.imports

        # Combine imports and code using the new utility function
        runnable_code = combined_code(code_solution)

        # Check imports
        test_results = []
        try:
            exec_as_main(imports)
        except Exception as e:
            self.log("---CODE IMPORT CHECK: FAILED---")
            # TODO update this if it becomes a problem, for now keep it simple. Maybe want to just unify it with the code flow below.
            test_result_message = f"""Your solution failed the import test. Here is the error: {e}."""
            return {
                "last_test_results": test_result_message,
                "error": "yes",
            }
        # Check execution
        # Use a shared scope for exec
        num_test_cases = len(test_cases)
        succeeded = 0
        for test_case in test_cases:
            global_scope = {
                "global_input": test_case["inputs"],
                "global_output": "",
                "debug_output": ""
            }
            if self.code_executor:
                import asyncio
                print("Running code executor...")
                executor_result = asyncio.run(self.code_executor(runnable_code, global_scope, self.additional_params))
                if "error" in executor_result:
                    test_results.append(f"""Failed test case.
Intermediate test case output: {global_scope["global_output"] if global_scope["global_output"] else "EMPTY"}
Final test case output: {executor_result["error"]}
Expected test case output:
{test_case['outputs']}
Debug test case output:
{global_scope['debug_output']}""")
                    continue
                if ("postprocessed_output" in executor_result):
                    final_output = executor_result["postprocessed_output"]
                    intermediate_output = global_scope["global_output"]
                else:
                    final_output = executor_result["global_output"]
                    intermediate_output = ""
            else:
                executor_result = exec_as_main(runnable_code, global_scope)
                final_output = global_scope["global_output"]
                intermediate_output = ""
                # TODO: Add a codegen for HTML parsing to be tacked on at the end once we have the final page.
            output_match, differences = self.diff_output(final_output, test_case["outputs"])
            if output_match:
                test_results.append(
                    f"Correct test case! Your output matched the expected output: {test_case['outputs']}. Successful debug output in case it's useful: {global_scope['debug_output']}")
                succeeded += 1
            else:
                # TODO(opt): Do we need test case input? It is rather large...maybe an LLM summary of test input?
                test_results.append(
                    f"""Failed test case.
Intermediate test case output: {intermediate_output if intermediate_output else "EMPTY"}
Final test case output: {final_output if final_output else "EMPTY"}
Expected test case output:
{test_case['outputs']}
Debug test case output:
{global_scope['debug_output']}
Differences:
{differences}""")
        if succeeded == num_test_cases:
            self.log(
                f"=========== CODE BLOCK CHECK: SUCCEEDED in {iterations} iterations ===========")
            return {
                "error": "no",
            }
        pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
        self.log("---CODE BLOCK CHECK: FAILED---")
        responses = "\n".join(
            [f"<test case {i} begin >\n{r}\n</test case>" for i, r in enumerate(test_results)])
        test_result_message = f"""Incorrect submission.\nPass rate: {pass_rate}\nResults:\n{responses}"""
        self.log("Code:\n" + runnable_code)
        self.log("Test results:\n" + test_result_message)
        return {
            "last_test_results": test_result_message,
            "error": "yes",
        }

    def _print_event(self, event: dict, _printed: set, max_length=1500):
        current_state = event.get("dialog_state")
        if current_state:
            self.log(f"Currently in: ", current_state[-1])
        # Is this from state? If so it may not work..
        message = event.get("messages")
        if message:
            if isinstance(message, list):
                message = message[-1]
            if message.id not in _printed:
                msg_repr = message.pretty_repr(html=True)
                if len(msg_repr) > max_length:
                    msg_repr = msg_repr[:max_length] + " ... (truncated)"
                self.log(msg_repr)
                _printed.add(message.id)

    def generate_diff(self, old_value, new_value):
        # Split the strings into individual characters for a character-level diff
        diff = list(ndiff(old_value, new_value))
        # Filter to only include lines that start with '+ ' or '- ' to show differences
        filtered_diff = [line for line in diff if line.startswith('+ ') or line.startswith('- ')]
        return filtered_diff

    def char_diff(self, str1, str2):
        # Use difflib.ndiff to get the diff
        diff = list(ndiff(str1, str2))
        # Process the diff to group and format differing characters
        differences = []
        current_diff = []
        last_op = None

        for item in diff:
            op, char = item[0], item[2]
            if op in ('-', '+'):
                if op != last_op:
                    if current_diff:
                        differences.append(''.join(current_diff))
                        current_diff = []
                    last_op = op
                current_diff.append(op + char)
            else:
                if current_diff:
                    differences.append(''.join(current_diff))
                    current_diff = []
                last_op = None

        if current_diff:
            differences.append(''.join(current_diff))

        return ' '.join(differences)

    def diff_output(self, global_output, expected_output):
        """
        Compares the actual output from the code execution to the expected output from the test case.
        If the expected output is valid JSON, compares as JSON objects, ignoring field ordering and formatting.
        Otherwise, compares as strings.

        Args:
            global_output (str): The output from code execution.
            expected_output (str): The expected output from the test case.

        Returns:
            tuple: (bool, str) - True if outputs match, False otherwise, and differences as a string.
        """
        try:
            # Try to parse both outputs as JSON
            json_global_output = json.loads(global_output)
            json_expected_output = json.loads(expected_output)
            # Compare as JSON objects
            differences = deepdiff.DeepDiff(json_global_output, json_expected_output, ignore_order=True)
            if differences == {}:
                self.log("Comparing as JSON objects: Match")
                return True, ""
            else:
                detailed_differences = {}
                for key, value in differences.items():
                    if 'values_changed' in key:
                        for subkey, change in value.items():
                            old_value, new_value = change['old_value'], change['new_value']
                            # Generate character-level differences
                            char_diff = self.char_diff(old_value, new_value)
                            detailed_differences[subkey] = {
                                'old_value': old_value,
                                'new_value': new_value,
                                'character_diff': char_diff
                            }
                self.log(f"Comparing as JSON objects: No Match. Differences: {detailed_differences}")
                return False, str(detailed_differences)
        except json.JSONDecodeError as e:
            # If either output is not valid JSON, compare as strings
            if str(global_output) == str(expected_output):
                self.log("Comparing as strings: Match")
                return True, ""
            else:
                self.log(f"Comparing as strings: No Match. Exception: {e}")
                return False, f"Expected: {expected_output}, Got: {global_output}"


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
            self.log("---DECISION: FINISH---")
            return "end"
        else:
            self.log("---DECISION: RE-TRY SOLUTION---")
            return "retry"

    def _generate_graph_image(self):
        try:
            graph_image = self.graph.get_graph().draw_mermaid_png()
            with open("data/graph.png", "wb") as f:
                f.write(graph_image)
            self.log("Graph saved as 'data/graph.png'.")
        except Exception as e:
            self.log(f"Failed to save graph: {e}")
    def log(self, message: str, filename="data/langchain_trace.txt"):
        mode = 'a' if self.log_file_initialized else 'w'
        print(message)
        with open(filename, mode) as file:
            file.write(str(message) + "\n")
        self.log_file_initialized = True

    def generate_code(self, user_prompt, test_cases):
        self.system_prompt_tuple = (
            "system", f"Overall task to solve: {user_prompt}")
        _printed = set()
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                # Checkpoints are accessed by thread_id
                "thread_id": thread_id,
            },
            "recursion_limit": 50
        }
        events = self.graph.stream(
            {"previous_corrections": [], "iterations": 0, "test_cases": test_cases}, config, stream_mode="values"
        )
        for event in events:
            self._print_event(event, _printed)
        return event['code_solution']

import json

if __name__ == "__main__":
    # Global variable to track if the log file has been cleared
    log_file_initialized = False
    import generalize_actions
    import web_voyager.tasks.tennis_task as tennis_task
    load_dotenv()
    with open("web_voyager/data/actions_log.json", "r") as f:
        action_log = json.load(f)
    
    # Remove parent_html from each action's params
    for action in action_log:
        if 'params' in action and 'parent_html' in action['params']:
            del action['params']['parent_html']

    params = {"date": "08/13/2024",
                          "interval": "30", "timeFrom": "12:00pm", "timeTo": "2:00pm"}

    # Next step: add playwright executor that runs the code and prints the final result as well.
    expected_output = {"Tennis" : ["11:30", "12:00", "1:00pm", "1:30pm", "2:00pm"]}
    test_cases = [{'inputs': {"params" : params, "action_log" : action_log}, 'outputs': expected_output}]
    code_generator = CodeGenerator()
    code_generator.set_code_executor(generalize_actions.execute_code)
    code_generator.set_additional_params({"input_prompt": tennis_task.get_prompt(), "output_format_example": expected_output})
    prompt = generalize_actions.get_prompt(test_cases)
    print(f"Prompt: {prompt}")
    code_solution = code_generator.generate_code(prompt, test_cases)
    code_solution = combined_code(code_solution)    
    with open("generalizer/out/succesful_func.txt", "w") as f:
        f.write(code_solution)
