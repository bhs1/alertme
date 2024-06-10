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


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    # Is this from state? If so it may not work..
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


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

        # Select LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.code_gen_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """
         You are an expert programmer. Ensure any code you provide can be executed with all
         required imports and variables defined. Structure your answer: 1) a prefix describing
         the code solution, 2) the imports, 3) the functioning code block.
         - The imports should only contain import statements. E.g. import math, import sys, etc., not actual code.

         - Instead of input from stdin, assume input is provided as an immutable string global variable "global_input".
         - Be sure that your solution is modular so that it's not dependent on the gloabl_input variable, that should be
         extracted in a separate function. DO NOT HARDCODE global_input in your solution, you must extract is from global global_input.
         - Likewise, instead of printing to stdout, return the result as a string global variable "global_output".
         Storing output in global_output should also be separated from the core business logic function.
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

        print(
            "---REFLECTING ON ERRORS, DEBUG STATEMENTS, and FORMULATING IMPROVEMENT PLAN---")
        # Ensure only the last three corrections are used, or fewer if less than three exist
        last_three_corrections = previous_corrections[-3:] if len(previous_corrections) >= 3 else previous_corrections
        context = f"""\
    Previous Corrections, Reflections and Code diffs:\n\n {self.correction_list_to_string(last_three_corrections)}
        """
        prompt = f"""
    (1) Determine what went wrong in each of the following test results: {last_test_results}.
    How does the actual output diff from the expected output and why do you think that is?
    Explain for each failing test case. If useful, consider the debugoutput when reasoning about your response.
     
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

        print("---GENERATING CODE SOLUTION---")
        # TODO(P2): Cap lookback at previous solutions to 3 as shown to work in the AlphaCodium paper.

        # Correction includes reflection, code diff
        iterations = state["iterations"]
        error = state["error"]
        reflection_prompt = state["reflection_prompt"]
        reflection_summary = state["reflection_summary"]
        messages = [self.system_prompt_tuple]
        if (error == "yes"):
            print("--------- Rewriting code with print statements -----------")
            messages += [
                ("user", f"""We just gave you this prompt:\n\n {reflection_prompt} \n\n and as a result \
        of the above prompt you gave this relfection summary:\n\n {reflection_summary}\n
        Main directive: Rewrite the code to fix the test cases. Be sure to follow the plan \
        in the reflection summary.
                 """)]

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

        print("---CHECKING CODE---")

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
            print("---CODE IMPORT CHECK: FAILED---")
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
                if str(global_scope["global_output"]) == str(test_case["outputs"]):
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
                {global_scope['debug_output']}""")
            except Exception as e:
                test_results.append(f"""Failed test case.
                            Actual test case output:
                            Exception {e}
                            Expected test case output:
                            {test_case['outputs']}
                            Debug test case output:
                            {global_scope['debug_output']}""")
            pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
        if succeeded == num_test_cases:
            print(
                f"=========== CODE BLOCK CHECK: SUCCEEDED in {iterations} iterations ===========")
            return {
                "error": "no",
            }
        print("---CODE BLOCK CHECK: FAILED---")
        responses = "\n".join(
            [f"<test case {i} begin >\n{r}\n</test case>" for i, r in enumerate(test_results)])
        test_result_message = f"""Incorrect submission.\nPass rate: {succeeded}/{num_test_cases}\nResults:\n{responses}"""
        return {
            "last_test_results": test_result_message,
            "error": "yes",
        }


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
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            return "retry"

    def _generate_graph_image(self):
        try:
            graph_image = self.graph.get_graph().draw_mermaid_png()
            with open("data/graph.png", "wb") as f:
                f.write(graph_image)
            print("Graph saved as 'data/graph.png'.")
        except Exception as e:
            print(f"Failed to save graph: {e}")

    def generate_code(self, user_prompt, test_cases):
        self.system_prompt_tuple = (
            "system", f"Overall task to solve: {user_prompt}")
        _printed = set()
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                # Checkpoints are accessed by thread_id
                "thread_id": thread_id,
            }
        }
        events = self.graph.stream(
            {"previous_corrections": [], "iterations": 0, "test_cases": test_cases}, config, stream_mode="values"
        )
        for event in events:
            _print_event(event, _printed)
        return event['code_solution']


if __name__ == "__main__":
    from prepare_request import get_request_params_map, get_request_replace_func, replace_fields
    load_dotenv()
    with open("data/request.txt", "r") as f:
        request = f.read()
    with open("data/request_output.txt", "r") as f:
        expected_output = f.read()
    #request_params_map = get_request_params_map(request)
    #print(request_params_map)
    # Got this from above.
    REQUEST_PARAMS_MAP = {'date': '06/15/2024',
                          'interval': '60', 'timeFrom': '45', 'timeTo': '60'}
    test_cases = [{'inputs': str(
        REQUEST_PARAMS_MAP) + "\n\n" + request, 'outputs': expected_output}]
    #func_str = get_request_replace_func(REQUEST_PARAMS_MAP, test_cases)
    with open("data/succesful_func.txt", "r") as f:
        func_str = f.read()
    print("Output:\n", replace_fields(request, REQUEST_PARAMS_MAP, func_str))
    

