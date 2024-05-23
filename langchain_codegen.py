import getpass
import os
import multiprocessing
import queue
import subprocess
import sys
import time
import traceback


###### Environment Variables ######
def _get_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_get_env("OPENAI_API_KEY")
# Recommended
_get_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
###### End Environment Variables ######


# Set the start method for multiprocessing to 'fork', enforcing it even if another method was set.
multiprocessing.set_start_method("fork", force=True)

# A function that executes a program in a separate process and communicates the result back via a queue.
def exec_program(q, program, input_data, expected_output, timeout):
    try:
        start_time = time.time()  # Record the start time to measure execution time.
        # Start a subprocess that runs the Python interpreter on the given program code.
        process = subprocess.Popen(
            [sys.executable, "-c", program],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Pass input data to the subprocess and allow it to run until it completes or the timeout is reached.
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
        # Check if the actual execution time exceeded the allowed timeout.
        if time.time() - start_time > timeout:
            raise TimeoutError("Execution timed out.")
        # Check the return code of the process to determine if it exited successfully.
        if process.returncode != 0:
            q.put(f"failed: {stderr}")
        else:
            # Compare the actual output from the subprocess with the expected output.
            if stdout.strip() == expected_output.strip():
                q.put("passed")
            else:
                q.put(f"wrong answer. Expected '{expected_output}', got '{stdout}'")
    except subprocess.TimeoutExpired:
        process.kill()
        q.put("timed out")
    except Exception as e:
        q.put(f"failed: {traceback.format_exc()}")  # Capture and return any exceptions that occur during execution.

# Function to set up the environment and handle the multiprocessing necessary to run the exec_program function.
def check_correctness(
    program: str, input_data: str, expected_output: str, timeout: float
) -> str:
    q = multiprocessing.Queue()  # Create a queue to communicate results from the subprocess.
    # Set up and start a new process to run the exec_program function.
    process = multiprocessing.Process(
        target=exec_program, args=(q, program, input_data, expected_output, timeout)
    )
    process.start()
    process.join(timeout=timeout + 1)  # Wait for the process to finish with an extra second as a buffer.
    if process.is_alive():
        process.terminate()  # Terminate the process if it is still running after the timeout.
        process.join()
        result = "timed out"
    else:
        try:
            result = q.get_nowait()  # Attempt to get the result from the queue.
        except queue.Empty:
            result = "no result returned"  # Handle the case where no result was placed in the queue.
    return result

from typing import Annotated, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class TestCase(TypedDict):
    inputs: str
    outputs: str


class State(TypedDict):
    # Append-only chat memory so the agent can try to recover from initial mistakes.
    messages: Annotated[list[AnyMessage], add_messages]
    # From the dataset. These are used for testing.
    test_cases: list[TestCase]
    runtime_limit: int
    status: str

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


class writePython(BaseModel):
    """Write python code that resolves the problem."""

    reasoning: str = Field(..., description="Conceptual solution.")
    pseudocode: str = Field(..., description="Detailed English pseudocode.")
    code: str = Field(..., description="Valid Python 3 solution to the problem")


class Solver:
    def __init__(self, llm: BaseChatModel, prompt: ChatPromptTemplate):
        self.runnable = prompt | llm.bind_tools([writePython])

    def __call__(self, state: State) -> dict:
        # Our agent only can see the "messages" and will ignore the test info
        return {"messages": [self.runnable.invoke({"messages": state["messages"]})]}

from langchain import hub
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo") # Update to GPT 4 for production.

from langchain_core.prompts import ChatPromptTemplate

# Define the template for the system message directly using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    messages = [
        ("system", """You are a world-class competitive programmer.
        Please reply with a Python 3 solution to the problem below. 
        First, reason through the problem and conceptualize a solution.
        Then write detailed pseudocode to uncover any potential logical errors or omissions.
        Finally output the working Python code for your solution, ensuring to fix any errors 
        uncovered while writing pseudocode. 

        No outside libraries are allowed. Be sure to separate any reading of stdin or stdout
        from your code so that the actual business logic is in its own function and clearly marked
        as the solution with a comment. So you should never call input() inside the function but rather
        call input() above the function and pass the result of input() to the function.
         
        {examples}"""),
        ("placeholder", "{messages}")
    ]
).partial(examples="")

solver = Solver(llm, prompt)


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# This is the node we will add to the graph.
def format_tool_message(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response + "\nMake all fixes using the writePython tool.",
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def evaluate(state: State):
    test_cases = state["test_cases"]
    runtime_limit = state["runtime_limit"]
    ai_message: AIMessage = state["messages"][-1]
    if not ai_message.tool_calls:
        return {
            "messages": [
                HumanMessage(
                    content="No code submitted. Please try again using the correct python code."
                )
            ]
        }
    try:
        code = ai_message.tool_calls[0]["args"]["code"]
    except Exception as e:
        return {"messages": [format_tool_message(repr(e), ai_message)]}
    num_test_cases = len(test_cases)
    succeeded = 0
    test_results = []
    for test_case in test_cases:
        input_data = test_case["inputs"]
        expected_output = test_case["outputs"]
        test_result = check_correctness(code, input_data, expected_output, runtime_limit)
        test_results.append(test_result)
        if test_result == "passed":
            succeeded += 1
    pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
    if pass_rate == 1:
        print("Code execution successful. Here is the code:")
        print(code)
        return {"status": "success"}

    responses = "\n".join(
        [f"<test id={i}>\n{r}\n</test>" for i, r in enumerate(test_results)]
    )
    response = f"Incorrect submission. Please respond with updated code.\nPass rate: {succeeded}/{num_test_cases}\nResults:\n{responses}"
    formatted_message = format_tool_message(response, ai_message)
    return {"messages": [formatted_message]}

from langgraph.graph import END, StateGraph

builder = StateGraph(State)
builder.add_node("solver", solver)
builder.set_entry_point("solver")
builder.add_node("evaluate", evaluate)
builder.add_edge("solver", "evaluate")


def control_edge(state: State):
    if state.get("status") == "success":
        return END
    return "solver"


builder.add_conditional_edges("evaluate", control_edge, {END: END, "solver": "solver"})
graph = builder.compile()

from IPython.display import Image

try:
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("data/graph.png", "wb") as f:
        f.write(graph_image)
    print("Graph saved as 'data/graph.png'.")
except Exception as e:
    print(f"Failed to save graph: {e}")

DESCRIPTION = """Create a function that sums two integers

INPUT FORMAT (input arrives from the terminal / stdin): The numbers to sum are separated
by spaces

OUTPUT FORMAT (output sent to the terminal / stdout): The sum of the two integers

Please pay attention to the system message above to see how to format your response.
"""


input_states = [
    {
        "messages": [("user", DESCRIPTION)],
        "test_cases": [
            {"inputs": "1 2", "outputs": "3"},
            {"inputs": "3 4", "outputs": "7"},
            {"inputs": "0 0", "outputs": "0"},
            {"inputs": "-1 -1", "outputs": "-2"},
            {"inputs": "100 200", "outputs": "300"}
        ],
        "runtime_limit": 10,
        "status": "in_progress",
        "problem_level": 0,
    }
]

input_state = input_states[0].copy()
# We will reduce the test cases to speed this notebook up
input_state["test_cases"] = input_state["test_cases"][:3]

from langchain_core.messages import BaseMessage
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client


# We don't need to include all the test cases in our traces.
def _hide_test_cases(inputs):
    copied = inputs.copy()
    # These are tens of MB in size. No need to send them up
    copied["test_cases"] = f"..."
    return copied


client = Client(hide_inputs=_hide_test_cases, hide_outputs=_hide_test_cases)
with tracing_v2_enabled(client=client):
    events = graph.stream(input_state)
    for event in events:
        for value in event.values():
            messages = value.get("messages")
            if messages:
                if isinstance(messages, list):
                    messages = value["messages"][-1]
                print(
                    "Assistant:",
                    str(messages.content).replace("\n", "\\n")[:1000],
                )
