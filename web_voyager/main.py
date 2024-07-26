# Optional: add tracing to visualize the agent trajectories
import os

from typing import List, Optional, TypedDict

from langchain_core.messages import SystemMessage
from playwright.async_api import Page
    
import asyncio
import platform
import sys
    
import base64

from langchain_core.runnables import chain as chain_decorator

    
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
sys.path.insert(0, '/Users/bensolis-cohen/Projects/alertme')
from langgraph_utils import generate_graph_image

import re

from langchain_core.runnables import RunnableLambda

from langgraph.graph import END, StateGraph

from prompt import web_voyager_prompt

import logging

import asyncio
from asyncio import DefaultEventLoopPolicy, SelectorEventLoop

# Ensure the logs directory exists
log_dir = "/Users/bensolis-cohen/Projects/alertme/web_voyager/logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join(log_dir, 'info.txt'),
                    filemode='a')

os.environ["LANGCHAIN_PROJECT"] = "Web-Voyager-3"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str
    outerHTML: str


class Prediction(TypedDict):
    action: str
    thought: Optional[str]
    args: Optional[List[str]]

async def compress_screenshot(screenshot, scale_factor: int = 1) -> str:
    """Compress the screenshot by reducing its width and height by 4x."""
    image = Image.open(io.BytesIO(screenshot))
    
    new_size = (image.width // scale_factor, image.height // scale_factor)
    compressed_image = image.resize(new_size, Image.LANCZOS)
    
    # Convert back to base64
    buffer = io.BytesIO()
    compressed_image.save(buffer, format="PNG")  # Save as PNG to avoid JPEG compression
    compressed_b64_image = base64.b64encode(buffer.getvalue()).decode()
    
    return compressed_b64_image

# This represents the state of the agent
# as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[str]
    observation: str  # The most recent response from a tool
    unannotated_img: str
    action_summary: str
    step: int
    formatted_scratchpad: str


async def click(state: AgentState):
    
    try:
        page = state["page"]
        click_args = state["prediction"]["args"]
        if click_args is None or len(click_args) != 1:
            return f"Failed to click bounding box labeled as number {click_args}"
        bbox_id = int(click_args[0])
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        await page.mouse.click(x, y)
        await page.wait_for_load_state('networkidle')
        return f"Clicked {bbox_id}"
    except Exception as e:
        return f"Error in click: {str(e)}"


async def type_text(state: AgentState):
    try:
        page = state["page"]
        type_args = state["prediction"]["args"]
        if type_args is None or len(type_args) != 2:
            return f"Failed to type in element from bounding box labeled as number {type_args}"
        bbox_id = int(type_args[0])
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        text_content = type_args[1]
        await page.mouse.click(x, y)
        select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
        await page.keyboard.press(select_all)
        await page.keyboard.press("Backspace")
        await page.keyboard.type(text_content)
        await page.keyboard.press("Enter")
        await page.wait_for_load_state('networkidle')  # Wait for the page to reach the networkidle state
        return f"Typed {text_content} and submitted"
    except Exception as e:
        return f"Error in type_text: {str(e)}"


async def scroll(state: AgentState):
    try:
        page = state["page"]
        scroll_args = state["prediction"]["args"]
        if scroll_args is None or len(scroll_args) != 2:
            return "Failed to scroll due to incorrect arguments."

        target, direction = scroll_args
        scroll_amount = 500 if target.upper() == "WINDOW" else 200
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount

        if target.upper() == "WINDOW":
            await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        else:
            target_id = int(target)
            bbox = state["bboxes"][target_id]
            x, y = bbox["x"], bbox["y"]
            await page.mouse.move(x, y)
            await page.mouse.wheel(0, scroll_direction)

        return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"
    except Exception as e:
        return f"Error in scroll: {str(e)}"


async def wait(state: AgentState):
    try:
        sleep_time = 5
        await asyncio.sleep(sleep_time)
        return f"Waited for {sleep_time}s."
    except Exception as e:
        return f"Error in wait: {str(e)}"


async def go_back(state: AgentState):
    try:
        page = state["page"]
        await page.go_back()
        return f"Navigated back a page to {page.url}."
    except Exception as e:
        return f"Error in go_back: {str(e)}"


async def to_google(state: AgentState):
    try:
        page = state["page"]
        await page.goto("https://www.google.com/")
        return "Navigated to google.com."
    except Exception as e:
        return f"Error in to_google: {str(e)}"
    
async def to_url(state: AgentState):
    try:
        page = state["page"]
        url = state["prediction"]["args"][0]
        await page.goto(url)
        return f"Navigated to {url}."
    except Exception as e:
        return f"Error in to_url: {str(e)}"

# Some javascript we will run on each step
# to take a screenshot of the page, select the
# elements to annotate, and add bounding boxes
with open("/Users/bensolis-cohen/Projects/alertme/web_voyager/mark_page.js") as f:
    mark_page_script = f.read()

@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            unannotated_page = await take_screenshot(page)
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
        "unannotated_img": unannotated_page,
        "bboxes": bboxes,
    }


async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for bbox in state["bboxes"]:
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        #outer_html = bbox.get("outerHTML", "")
        scrollable = bbox.get("isScrollable", False)
        index = bbox.get("index", "")
        labels.append(f'{index} (<{el_type}/>): "{text}" Scrollable: {scrollable}')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}    


def parse(text: str) -> dict:
    action_prefix = "Action: "
    thought_prefix = "Thought: "
    
    lines = text.strip().split("\n")
    
    if not lines[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    
    action_block = lines[-1]
    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    
    thought = None
    for line in lines:
        if line.startswith(thought_prefix):
            thought = line[len(thought_prefix):].strip()
            break
    
    return {"action": action, "args": action_input, "thought": thought}


# Will need a later version of langchain to pull
# this image prompt template
prompt = web_voyager_prompt
llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)

async def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad", list())    
    step = state.get("step")
    if step is None:
        step = 1
    
    thought = state['prediction'].get('thought', '')
    latest_step = f"""Action attempted: {state['prediction']['action']} with args: {state['prediction'].get('args', {})}

Action Result: {state['action_summary']}
    """
    print("================== STEP ====================")
    print(latest_step)
    old.append(latest_step)
    if len(old) > 3:
        old = old[-3:]  # Keep only the latest 3 steps
    
    step += 1
    formatted_scratchpad = format_scratchpad(old)

    return {**state, "scratchpad": old, "formatted_scratchpad": formatted_scratchpad, "step": step}

def format_scratchpad(scratchpad):
    content = "\n----------------------------------\n".join([f"{i+1}. {entry}" for i, entry in enumerate(scratchpad)])
    return "======================================================\n" + content + "\n======================================================"

agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

from PIL import Image
import io

async def take_screenshot(page):
    screenshot = await page.screenshot()
    compressed_screenshot = await compress_screenshot(screenshot)
    return compressed_screenshot

from prompt import compare_screenshots_prompt

async def reflect(state: AgentState):
    """
    (1) Take a screenshot of the page
    (2) Compare it to the previous screenshot before the action
    (3) Summarize the result of the action and whether it succeeded or failed
    (4) Append any error to the summary
    (5) Return the summary in the agent state
    """

    page = state["page"]
    prediction = state.get("prediction")
    action_taken = f"Action attempted: {prediction['action']} with args: {prediction.get('args', {})}. Which results in:"
    thought = prediction.get("thought")
    if thought:
        action_taken = f"Thought: {thought}\n" + action_taken
    old_screenshot = state.get("img")
    #new_screenshot = await take_screenshot(page)  # Take a new screenshot of the page
    annotated_state = await annotate({"page": page})  # Call annotate on the new screenshot
    new_screenshot = annotated_state["img"]
    
    # Save the images to files in the scratch directory
    def save_image_to_file(image_data, filename):
        file_path = os.path.join("/Users/bensolis-cohen/Projects/alertme/web_voyager/scratch", filename)
        with open(file_path, "wb") as file:
            file.write(image_data.encode('utf-8'))
    
    save_image_to_file(new_screenshot, "new_screenshot.jpg")
    save_image_to_file(old_screenshot, "old_screenshot.jpg")
    
    llm = compare_screenshots_prompt | ChatOpenAI(
            model="gpt-4o", temperature=0)
    comparison_result = llm.invoke({"new_screenshot": new_screenshot, "action_taken": action_taken, "old_screenshot": old_screenshot, "input": state["input"]}, {"run_name": "ReflectOnDiff", "tags" : ["reflect-on-diff"]})        
    
    summary = f"Action result: {comparison_result.content}"

    if "error" in state:
        summary += f"\nError: {state['error']}"

    state["action_summary"] = summary
    return state

graph_builder = StateGraph(AgentState)


graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")

graph_builder.add_node("reflect", RunnableLambda(reflect))
graph_builder.add_edge("reflect", "update_scratchpad")

graph_builder.add_node("update_scratchpad", RunnableLambda(update_scratchpad))
graph_builder.add_edge("update_scratchpad", "agent")

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
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    graph_builder.add_edge(node_name, "reflect")


def select_tool(state: AgentState):
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action


graph_builder.add_conditional_edges("agent", select_tool)

graph = graph_builder.compile()
generate_graph_image(graph, filename="/Users/bensolis-cohen/Projects/alertme/web_voyager/logs/web_voyager_graph.png")

from playwright.async_api import async_playwright

async def setup_browser():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    await page.set_viewport_size({"width": 1700, "height": 900})  # Set the viewport size
    _ = await page.goto("https://www.google.com")
    return page

async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": list(),
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    
    async for event in event_stream:
        if "agent" not in event:
            continue
        pred = event["agent"].get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")        
        
        if "ANSWER" in action:
            final_answer = action_input[0]
            break
    return final_answer

async def ask_agent(question: str, page):
    res = await call_agent(question, page)
    print(f"Question: {question}")
    return res

async def main():
    page = await setup_browser()

    #await ask_agent("Could you explain the WebVoyager paper (on arxiv)?", page)
    #await ask_agent("Navigate to amazon.com and buy the cheapest microwave oven on Amazon under $50. Add it to cart and proceed to checkout.", page)
    res = await ask_agent("""Go to url: https://gtc.clubautomation.com/. Use username: bensc77, password: wjbr8KLh6t6NCm.
Task: Search for courts that satisfy the criteria (do not reserve):
 - date: 07/26/2024
 - start_time: 10am
 - end_time: 9pm
 - duration: 60 mins.
""", page)
    #await ask_agent("Could you check google maps to see when i should leave to get to SFO by 7 o'clock? starting from SF downtown.", page)
    
    print(f"Final response: {res}")

if __name__ == "__main__":

    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
        sys.stderr = original_stderr