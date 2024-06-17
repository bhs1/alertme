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
import deepdiff

# Data model

class Code(BaseModel):
    """Code output"""
    correction_summary: str = Field(
        description="Summary of what was corrected to produce this code.")
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
    return f"{code_solution.imports}\n{code_solution.code}"


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
                 """
    Coding agent general guideance:
         - You are an expert programmer. Ensure any code you provide can be executed with all
         required imports and variables defined. Structure your answer: 1) a prefix describing
         any corrections, 2) the imports, 3) the functioning code block. The imports should only
         contain import statements. E.g. import math, import sys, etc., not actual code.
         - Note that you will be outputing this in JSON format so all strings especially in the code block
         must be escaped.
         
    I/O handling:
         - Instead of input from stdin, assume input is provided as an immutable string global variable "global_input".
         - Be sure that your solution is modular so that it's not dependent on the gloabl_input variable, that should be
         extracted in a separate function. DO NOT HARDCODE global_input in your solution, you must extract is from global global_input.
         - Likewise, instead of printing to stdout, return the result as a string global variable "global_output".
         Storing output in global_output should also be separated from the core business logic function.

    Other guideance:         
         - IMPORTANT: For malformed input, instead of validating the input and catching exceptions, you should try to
         update your code in order to parse the input as best you can. E.g. if there is a json parsing error, try to
         update the code to parse the input as a string first and transform it to be valid json.
         Here is the user question:
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
    
    Note: To save resource, try to avoid printing entire HTTP requests or json, instead only print parts that may be useful for debugging \
    you may wish to summarize the structure in plain text the parts that you choose leave out.
     
    (2) Construct an improvement plan to fix the test failures. When formulating this plan, observe the previous solutions
    shown above and try not to repeat fixes that we already know do not lead to a fix.
        
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
            exec(imports)
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
            try:
                exec(runnable_code, global_scope)
                output_match, differences = self.check_code(global_scope["global_output"], test_case["outputs"])
                if output_match:
                    test_results.append(
                        f"Correct test case! Your output matched the expected output: {test_case['outputs']}. Successful debug output in case it's useful: {global_scope['debug_output']}")
                    succeeded += 1
                else:
                    # TODO(opt): Do we need test case input? It is rather large...maybe an LLM summary of test input?
                    test_results.append(
                        f"""Failed test case.
            Actual test case output:
            {global_scope['global_output'] if global_scope['global_output'] else "EMPTY"}
            Expected test case output:
            {test_case['outputs']}
            Debug test case output:
            {global_scope['debug_output']}
            Differences:
            {differences}""")
            except Exception as e:
                test_results.append(f"""Failed test case.
            Actual test case output:
            Exception {e}
            Expected test case output:
            {test_case['outputs']}
            Debug test case output:
            {global_scope['debug_output']}""")
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
    def check_code(self, global_output, expected_output):
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
                self.log(f"Comparing as JSON objects: No Match. Differences: {differences}")
                return False, str(differences)
        except json.JSONDecodeError as e:
            # If either output is not valid JSON, compare as strings
            if str(global_output) == str(expected_output):
                self.log("Comparing as strings: Match")
                return True, ""
            else:
                self.log(f"Comparing as strings: No Match. Exception: {e}")
                return False, f"Expected: {expected_output}, Got: {global_output}"


    def validate_test_cases_json(test_cases: List[TestCase]) -> bool:
        """
        Validates that all 'inputs' in the test cases are valid JSON.

        Args:
            test_cases (List[TestCase]): List of test cases to validate.

        Returns:
            bool: True if all 'inputs' are valid JSON, False otherwise.
        """
        import json
        for test_case in test_cases:
            try:
                self.log("validating inputs is json:\n" + test_case['inputs'])
                json.loads(test_case['inputs'])
            except json.JSONDecodeError:
                    return False
            return True

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
    from prepare_request import get_request_params_map, combine_request_data, get_request_replace_func, replace_fields
    load_dotenv()
    with open("data/request.json", "r") as f:
        request = json.load(f)
    with open("data/request_output.json", "r") as f:
        expected_output = f.read()
    #request_params_map = get_request_params_map(request)
    #print(request_params_map)
    # Got this from above.
    REQUEST_PARAMS_MAP = {"date": "06/25/2024",
                          "interval": "30", "timeFrom": "0", "timeTo": "24"}
    # Combine the data    
    combined_data = combine_request_data(REQUEST_PARAMS_MAP, request)
    test_cases = [{'inputs': combined_data, 'outputs': expected_output}]
    func_str = get_request_replace_func(test_cases)
    #with open("data/succesful_func.txt", "r") as f:
    #    func_str = f.read()
    print(replace_fields(request, REQUEST_PARAMS_MAP, func_str))

