import sys

sys.path.insert(0, '.')
sys.path.insert(0, './web_voyager')

from dotenv import load_dotenv
from web_voyager.playwright_actions import setup_browser
from web_voyager.prompt import compare_screenshots_prompt
import json
from web_voyager.playwright_actions import click as playwright_click, type_text as playwright_type_text, scroll_until_visible as playwright_scroll, go_back as playwright_go_back, to_google as playwright_to_google, to_url as playwright_to_url
import logging
from web_voyager.prompt import web_voyager_prompt, param_conversion_code_prompt_template, task_param_used_prompt_template
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from web_voyager.utils import mask_sensitive_data, compress_screenshot, take_screenshot
from utils import save_image_to_file
from langgraph_utils import generate_graph_image
import os
import uuid
from langsmith import traceable



from typing import List, Optional, TypedDict

from playwright.async_api import Page

import asyncio

import base64

from langchain_core.runnables import chain as chain_decorator
from langchain_core.tracers.langchain import wait_for_all_tracers

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# TODO: Make this file object oriented instead of using a global page object.
global page

# Ensure the logs directory exists
log_dir = "./web_voyager/logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join(log_dir, 'info.txt'),
                    filemode='a')

logging.getLogger("httpx").setLevel(logging.WARNING)
#2024-08-20 12:10:37,978 - langsmith.client - WARNING - [Run 20240820_120917] client.py:1430 - [Run 20240820_120917] client.py:1430 - Failed to batch ingest runs: LangSmithError('Failed to POST https://api.smith.langchain.com/runs/batch in LangSmith API. HTTPError(\'403 Client Error: Forbidden for url: https://api.smith.langchain.com/runs/batch\', \'<!doctype html><meta charset="utf-8"><meta name=viewport content="width=device-width, initial-scale=1"><title>403</title>403 Forbidden\')')
logging.getLogger("langsmith.client").setLevel(logging.ERROR)

os.environ["LANGCHAIN_PROJECT"] = "Web-Voyager-3"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str
    outerHTML: str
    parentHTML: str
    isScrollable: bool
    isTypeable: bool
    isClickable: bool
    index: str
    xpath: str

class TaskParamCode(BaseModel):
    code_thought: str = Field(description="Thought: Your brief thoughts on how you plan to extract input_string to output_string.")
    code: str = Field(description="Code to extract input_string to output_string.")

class TaskParamSelection(BaseModel):
    task_param_used_explanation: Optional[str] = Field(description="task_param_used_explanation")
    task_param_used: str = Field(description="Name of the task_param used to fill the [Text] in action string.")

class Prediction(BaseModel):
    thought: Optional[str] = Field(description="Thought: Your brief thoughts.")
    raw_predicted_action_string: Optional[str] = Field(
        description="Prediction action string and args, formatted as described.")
    action: str = Field(description="One action you choose")
    args: Optional[List[str]] = Field(description="List of action args.")

class Reflection(BaseModel):
    """Reflection output"""
    intention_of_action: str = Field(
        description="Intention of the attempted action.")
    result_of_action: str = Field(
        description="Description of relevant details before screenshot, after screenshot, and comparison of the two.")
    explanation: str = Field(
        description="Brief explanation of why the action was a success, partial success, or failure.")
    success_or_failure_result: str = Field(
        description="Must be one of SUCCESS, PARTIAL_SUCCESS, or FAILURE.")
    action_summary: Optional[str] = Field(description="In a few words describe the action taken and the result.")


# This represents the state of the agent
# as it proceeds through execution


class AgentState(TypedDict):
    task: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: List[str]
    observation: str
    action_summary: str
    step: int
    formatted_scratchpad: str
    task_params: dict
    error: str
    task_param_code: str
    previous_action_summaries: List[str]
    reflection: Reflection
    task_param_used: str
    bbox_descriptions: str


actions_log = []


def store_action(action_name: str, params: dict):
    actions_log.append({"action": action_name, "params": params})

def update_previous_action_with_reflection(reflection_obj, task_param_code):    
    last_action = actions_log[-1]
    last_action["reflection"] = reflection_obj.dict()
    last_action["task_param_code"] = task_param_code

async def click(state: AgentState):
    try:
        global page
        click_args = state["prediction"].args
        task_param_used = state['task_param_used']
        if click_args is None or len(click_args) == 0 or len(click_args) == 3:
            # TODO(P2): Add more validation to click args.
            return f"Failed to click bounding box labeled as number {click_args}"
        bbox_id = int(click_args[0])
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        xpath = bbox.get("xpath", "")
        click_result = await playwright_click(page, x, y, bbox.get("text", ""))
        print(f"Clicked {bbox_id} with result {click_result}")        
        store_action("click", {"x": x, "y": y,
                     "bbox_id": bbox_id, "xpath": xpath, "click_result": click_result, "text": bbox.get("text", ""), "parent_html": bbox.get("parentHTML", ""), "task_param_used": task_param_used})
        return f"Clicked {bbox_id}"
    except Exception as e:
        return f"Error in click: {str(e)}"


async def type_text(state: AgentState):
    try:
        global page
        type_args = state["prediction"].args
        task_param_used = state['task_param_used']
        if type_args is None or len(type_args) != 2:
            return f"Failed to type in element from bounding box labeled as number {type_args}"
        bbox_id = int(type_args[0])
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        text_content = type_args[1]
        xpath = bbox.get("xpath", "")
        await playwright_type_text(page, x, y, text_content)
        store_action("type_text", {
                     "x": x, "y": y, "bbox_id": bbox_id, "text": text_content, "xpath": xpath, "task_param_used": task_param_used})
        return f"Typed {text_content} and submitted"
    except Exception as e:
        return f"Error in type_text: {str(e)}"

async def scroll(state: AgentState):
    try:
        global page
        scroll_args = state["prediction"].args
        task_param_used = state['task_param_used']
        if scroll_args is None or len(scroll_args) != 3:
            print(f"Failed to scroll due to incorrect arguments: {scroll_args}")
            return "Failed to scroll due to incorrect arguments."

        target, direction, text = scroll_args

        if target.upper() == "WINDOW":
            x, y = 0, 0
            xpath = ""
            scroll_amount = 500
        else:
            x, y = state["bboxes"][int(
                target)]["x"], state["bboxes"][int(target)]["y"]
            xpath = state["bboxes"][int(target)].get("xpath", "")
            scroll_amount = 200
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await playwright_scroll(page, x, y, direction, scroll_direction, text)
        store_action("scroll", {"x": x, "y": y, "direction": direction,
                     "scroll_direction": scroll_direction, "xpath": xpath, "text": text, "task_param_used": task_param_used})
        return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'} until {text} is visible."
    except Exception as e:
        print(f"Error in scroll: {str(e)}")
        return f"Error in scroll: {str(e)}"


async def wait(state: AgentState):
    try:
        sleep_time = 5
        await asyncio.sleep(sleep_time)
        store_action("wait", {"sleep_time": sleep_time})
        return f"Waited for {sleep_time}s."
    except Exception as e:
        return f"Error in wait: {str(e)}"


async def go_back(state: AgentState):
    try:
        global page
        await playwright_go_back(page)
        store_action("go_back", {})
        return f"Navigated back a page to {page.url}."
    except Exception as e:
        return f"Error in go_back: {str(e)}"


async def to_google(state: AgentState):
    try:
        global page
        await playwright_to_google(page)
        store_action("to_google", {})
        return "Navigated to google.com."
    except Exception as e:
        return f"Error in to_google: {str(e)}"


async def to_url(state: AgentState):
    try:
        global page
        url = state["prediction"].args[0]
        await playwright_to_url(page, url)
        store_action("to_url", {"url": url, "task_param_used": state['task_param_used']})
        return f"Navigated to {url}."
    except Exception as e:
        return f"Error in to_url: {str(e)}"

# Some javascript we will run on each step
# to take a screenshot of the page, select the
# elements to annotate, and add bounding boxes
with open("./web_voyager/mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            # May be loading...
            asyncio.sleep(3)
    screenshot = await take_screenshot(page)
    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")
    return {
        "img": screenshot,
        "bboxes": bboxes,
    }


async def annotate(state):    
    marked_page = await mark_page.with_retry().ainvoke(page)
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for bbox in state["bboxes"]:
        text = bbox.get("text", "").strip()
        scrollable = bbox.get("isScrollable", False)
        typable = bbox.get("isTypeable", False)
        clickable = bbox.get("isClickable", False)
        index = bbox.get("index", "")

        available_actions = []
        if scrollable:
            available_actions.append(f"Scroll")
        if clickable:
            available_actions.append(f"Click")
        if typable:
            available_actions.append(f"Type")

        actions_str = ", ".join(available_actions)        

        labels.append(f'''(Box {index}) {actions_str}
    - Text: "{text}"
''')
    bbox_descriptions = "\nAvailable actions (choose one):\n" + "\n".join(labels)
    bbox_descriptions += """

Not associated with a box:
    Wait 
    GoBack
    Google
    ToUrl
    ANSWER
"""
    print("Calling LLM to determine action...")    
    return {**state, "bbox_descriptions": bbox_descriptions}


# Chains
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Add this near the top of the file, after other global variables
should_wait_for_all_tracers = True

# Updated function with timing logic
def wait_for_tracers():
    if should_wait_for_all_tracers:
        print("Waiting for traces to upload...")
        start_time = asyncio.get_event_loop().time()
        wait_for_all_tracers()
        end_time = asyncio.get_event_loop().time()
        print(f"Traces uploaded. Time taken: {end_time - start_time:.2f} seconds")
    else:
        print("Skipping trace upload wait.")

@traceable(name="agent_node")
async def agent_node(state: AgentState):
    state = await (RunnableLambda(annotate) | RunnableLambda(format_descriptions)).ainvoke(state)
    agent =  web_voyager_prompt | llm.with_structured_output(Prediction)
    prediction = await agent.ainvoke(state, {"tags": ["agent-node"]})        
    wait_for_tracers()
    return {**state, "prediction": prediction}

compare_chain = compare_screenshots_prompt | llm.with_structured_output(Reflection)

import web_voyager.tasks.tennis_task as tennis_task

async def setup(state: AgentState):
    global page
    page = await setup_browser()    
    return {**state}

async def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad", list())
    step = state.get("step")
    if step is None:
        step = 1

    args = state['prediction'].args or []
    bbox_id = None
    if args:
        try:
            bbox_id = int(args[0])
        except ValueError:
            bbox_id = None

    modified_args = args.copy()
    if bbox_id is not None:
        modified_args[0] = "Numerical_Label"

    bbox_text = state['bboxes'][bbox_id]['text'] if bbox_id is not None else "N/A"
    action_attempted = f"Action attempted: {state['prediction'].action} with args: {modified_args}"
    if bbox_text and bbox_text != "N/A":
        action_attempted += f", bounding box text: '{bbox_text}'"

    latest_step = f"""{action_attempted}

{state['action_summary']}
    """
    if state.get("error"):
        latest_step += f"\nError: {state['error']}"
        state["error"] = ""
    old.append(latest_step)
    if len(old) > 3:
        old = old[-3:]  # Keep only the latest 3 steps

    step += 1
    formatted_scratchpad = format_scratchpad(old)
    
    # previous_action_summaries = state.get('previous_action_summaries', [])
    # if not previous_action_summaries:
    #     previous_action_summaries = []

    # success_or_failure_result = state['reflection'].success_or_failure_result
    # action_summary = state['prediction'].action_summary + f", Result: ({success_or_failure_result})"
    # if action_summary:
    #     previous_action_summaries.append(action_summary)
    #formatted_scratchpad += "\nPrevious actions attempted (keep in mind that the actions may have been undone or failed, so check the screenshot to verify current state.):\n" + "\n".join(previous_action_summaries)

    return {**state, "scratchpad": old, "formatted_scratchpad": formatted_scratchpad, "step": step } # "previous_action_summaries" : previous_action_summaries}


def format_scratchpad(scratchpad):
    content = "\n----------------------------------\n".join(
        [f"{i+1}. {entry}" for i, entry in enumerate(scratchpad)])
    return "======================================================\n" + content + "\n======================================================"

def get_action_text(prediction: Prediction, state: AgentState):
    action = prediction.action
    args = prediction.args
    try:
        if action == "Type":
            return args[1]
        elif action == "ToUrl":
            return args[0]
        elif action == "Scroll":
            return args[2]
        elif action == "Click":
            bbox_id = int(args[0])
            bbox = state["bboxes"][bbox_id]
            return bbox["text"]
    except Exception as e:
        # TODO(P3): Feed this back to the agent to regenerate the action with correct args.
        print(f"Warning: Malformed action args: {str(e)}")
    return ""

@traceable(name="generate_task_param_used")
async def generate_task_param_used(state: AgentState):        
    prediction = state["prediction"]
    try:
        bbox_id = int(prediction.args[0]) if prediction.args else None
    except ValueError:
        bbox_id = None
    bbox_text = state['bboxes'][bbox_id]['text'] if bbox_id else "N/A"
    action_string = prediction.action + " " +  str(prediction.args)
    print("Predicting task_param_used...")
    chain = task_param_used_prompt_template | llm.with_structured_output(TaskParamSelection)
    task_param_response = await chain.ainvoke(
        {"raw_predicted_action_string": action_string,
         "action_thought": prediction.thought,
         "bbox_text": bbox_text,
        **state},
        {"tags": ["generate-task-param-used"]})
    task_param_used = task_param_response.task_param_used
    print(f"Predicted task_param_used: {task_param_used}")
    wait_for_tracers()
    return {**state, "task_param_used": task_param_used}

@traceable(name="generate_task_param_code")
async def generate_task_param_code(state: AgentState):
    prediction = state["prediction"]
    task_param_used = state["task_param_used"]
    if not task_param_used or task_param_used == "None":
        return {**state}
    if task_param_used not in state["task_params"]:
        # TODO(P3): Feed this back to the agent to regenerate the action with correct args.
        print(f"Warning: Task param {task_param_used} not found in task params {state['task_params']}. Replacing with error_task_param_used_not_found.")
        state['task_param_used'] = "error_task_param_used_not_found"
        return {**state}

    task_param_value = state["task_params"][task_param_used]
    action_text = get_action_text(prediction, state)
    if action_text == task_param_value or action_text == "":
        return {**state}
    
    chain = param_conversion_code_prompt_template | llm.with_structured_output(TaskParamCode)
    print(f"Generating task param code for {task_param_used} with value {task_param_value} and action text {action_text}")
    start_time_llm = asyncio.get_event_loop().time()
    task_param_code_obj = await chain.ainvoke({"input_string": task_param_value, "output_string": action_text}, {"tags": ["generate-task-param-code"]})
    end_time_llm = asyncio.get_event_loop().time()
    print(f"LLM call time: {end_time_llm - start_time_llm:.2f} seconds")
    
    wait_for_tracers()
    return {"task_param_code": task_param_code_obj.code}

@traceable(name="reflect")
async def reflect(state: AgentState):
    """
    (1) Take a screenshot of the page
    (2) Compare it to the previous screenshot before the action
    (3) Summarize the result of the action and whether it succeeded or failed
    (4) Append any error to the summary
    (5) Return the summary in the agent state
    """

    try:
        bbox_id = int(state['prediction'].args[0]
                      ) if state['prediction'].args else None
    except ValueError:
        bbox_id = None
    bbox_text = state['bboxes'][bbox_id]['text'] if bbox_id is not None else "N/A"
    if bbox_text and bbox_text != "N/A":
        print_str = f"\033[1mExecuted action: {state['prediction'].action} with args: {state['prediction'].args}, and task_param_used: {state['task_param_used']}, Bounding box text: {bbox_text}\033[0m"
    else:
        print_str = f"\033[1mExecuted action: {state['prediction'].action} with args: {state['prediction'].args}, and task_param_used: {state['task_param_used']}\033[0m"
    masked_print_str = await mask_sensitive_data(print_str)
    
    print(masked_print_str)    

    global page
    prediction = state.get("prediction")
    action_taken = f"Action attempted: {prediction.action} with args: {prediction.args}. Which results in:"
    thought = prediction.thought
    if thought:
        action_taken = f"Thought: {thought}\n" + action_taken
    old_screenshot = state.get("img")
    # new_screenshot = await take_screenshot(page)  # Take a new screenshot of the page
    # Call annotate on the new screenshot
    annotated_state = await annotate({"page": page})
    new_screenshot = annotated_state["img"]

    save_image_to_file(new_screenshot, "new_screenshot.jpg")
    save_image_to_file(old_screenshot, "old_screenshot.jpg")
    run_id = str(uuid.uuid4())
    print("Calling LLM to reflect on action...")
    start_time = asyncio.get_event_loop().time()
    comparison_result = await compare_chain.ainvoke({"new_screenshot": new_screenshot, "action_taken": action_taken, "old_screenshot": old_screenshot,
                                   "task": state["task"]}, {"tags": ["reflect-on-diff", prediction.action, run_id]})
    end_time = asyncio.get_event_loop().time()
    print(f"Time taken for comparison: {end_time - start_time:.2f} seconds")
    
    wait_for_tracers()

    explanation = comparison_result.explanation
    success_or_failure_result = comparison_result.success_or_failure_result
    summary = f"""Action result:
        {success_or_failure_result}
Explanation:
    {explanation}
"""
    if "FAILURE" in summary:
        print(f"\033[1;31m{summary}\033[0m")  # Bold red
    else:
        print(f"\033[1;32m{summary}\033[0m")  # Bold green
    print(f"Run ID: {run_id}")

    if "error" in state:
        summary += f"\nError: {state['error']}"

    
    state["action_summary"] = summary
    state["reflection"] = comparison_result
    update_previous_action_with_reflection(comparison_result, state["task_param_code"])
    state["task_param_code"] = "" # Clear the task param code
    return state


def select_tool(state: AgentState):
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"].action
    if "ANSWER" in action:
        return "finish"
    if action not in tools:
        state["error"] = f"Invalid action: {action}"
        return "update_scratchpad"
    return action

async def end(state: AgentState):
    res = state["prediction"].args[0] # TODO: Format this.
    masked_res = await mask_sensitive_data(res)
    print(f"Final response: {masked_res}")

    await save_actions_log()
    print("Waiting for remaining traces to upload...")
    wait_for_tracers()
    print("Ending...")

graph_builder = StateGraph(AgentState)


graph_builder.add_node("agent", agent_node)
graph_builder.add_node("generate_task_param_used", generate_task_param_used)
graph_builder.add_edge("agent", "generate_task_param_used")
# TODO: add self directed conditional edges for these in case validation fails.
graph_builder.add_edge("generate_task_param_used", "generate_task_param_code")
graph_builder.add_node("generate_task_param_code", generate_task_param_code)
graph_builder.add_node("reflect", RunnableLambda(reflect))

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
    "ToUrl": to_url,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {
            "observation": observation}),
    )
    graph_builder.add_edge(node_name, "reflect")

graph_builder.add_conditional_edges("generate_task_param_code", select_tool)

graph_builder.add_node("update_scratchpad", RunnableLambda(update_scratchpad))
graph_builder.add_edge("reflect", "update_scratchpad")
graph_builder.add_edge("update_scratchpad", "agent")

graph_builder.add_node("setup", setup)
graph_builder.add_node("finish", end)
graph_builder.add_edge("finish", END)
graph_builder.set_entry_point("setup")
graph_builder.add_edge("setup", "agent")

graph = graph_builder.compile()
generate_graph_image(
    graph, filename="./web_voyager/logs/web_voyager_graph.png")


async def call_agent(task: str, task_params: dict, max_steps: int = 300):
    config = {
        "recursion_limit": max_steps
    }
    await graph.ainvoke(
        {
            "scratchpad": list(),
            "task": task,
            "task_params": task_params
        },
        config
    )

# Load environment variables from .env file
load_dotenv()

async def save_actions_log():
    log_path = "./web_voyager/data/actions_log.json"
    with open(log_path, "w") as f:
        json.dump(actions_log, f, indent=4)


def run_web_voyager(task: str, task_params: dict, wait_for_tracers: bool = True):
    global should_wait_for_all_tracers
    should_wait_for_all_tracers = wait_for_tracers
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    asyncio.run(call_agent(task=task, task_params=task_params))
    sys.stderr = original_stderr

if __name__ == "__main__":
    run_web_voyager(task=tennis_task.get_prompt(), task_params=tennis_task.get_task_params(), wait_for_tracers=True)
