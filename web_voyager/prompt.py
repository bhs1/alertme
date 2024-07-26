from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.messages.tool import ToolMessage

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)
from langchain.schema import AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage
from typing import List, Union

# System message template
system_template = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will
feature Numerical Labels placed in the BOTTOM RIGHT corner of each Web Element. Carefully analyze the visual
information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow
the guidelines and choose one of the following actions:

1. Click a Web Element.
2. Delete existing content in a textbox and then type content.
3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. Pay attention!! The default scroll is the whole window. But if the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area using Numerical_Label.
 - Before choosing a number to scroll, be sure to check the Valid Bounding Boxes to see if the element is scrollable. If it is not, just scroll the nearest scrollable neighbor element. It should work.
4. Wait 
5. Go back
7. Return to google to start over.
8. Go to a specific URL
9. Respond with the final answer

Correspondingly, Action should STRICTLY follow the format:

- Click [Numerical_Label];
- Type [Numerical_Label]; [Content] 
- Scroll [Numerical_Label or WINDOW]; [up or down] e.g. Scroll [30]; down 
- Wait 
- GoBack
- Google
- ToUrl [URL]
- ANSWER; [content]

Key Guidelines You MUST follow:

* Action guidelines *
1) Execute only one action per iteration.
2) When clicking or typing, ensure to select the correct bounding box.
3) Numeric labels lie in the bottom-right corner of their corresponding bounding boxes and are colored the same.
4) Don't try the same action multiple times if it keeps failing.

* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages
2) Select strategically to minimize time wasted.

Your reply should strictly follow the format:

Thought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}
Action: {{One Action format you choose}}
Then the User will provide:
Observation: {{A labeled screenshot Given by User}}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human message template
image_prompt = ImagePromptTemplate(
    input_variables=["img"],
    template={"url": "data:image/png;base64,{img}"}
)

bbox_prompt = PromptTemplate(
    input_variables=["bbox_descriptions"],
    template="{bbox_descriptions}"
)

input_prompt = PromptTemplate(
    input_variables=["input"],
    template="{input}"
)

scratchpad_prompt = PromptTemplate(
    input_variables=["formatted_scratchpad"],
    template="\n\nMost recent actions:\n\n{formatted_scratchpad}"
)

human_message_prompt = HumanMessagePromptTemplate(prompt=[image_prompt, bbox_prompt, input_prompt, scratchpad_prompt])

# Chat prompt template
web_voyager_prompt = ChatPromptTemplate(
    input_variables=["bbox_descriptions", "img", "input", "formatted_scratchpad"],
    messages=[
        system_message_prompt,
        human_message_prompt
    ]
)

# Image prompt for new screenshot
new_screenshot_prompt = ImagePromptTemplate(
    input_variables=["new_screenshot"],
    template={"url": "data:image/png;base64,{new_screenshot}"}
)

# Image prompt for input annotated screenshot
old_screenshot_prompt = ImagePromptTemplate(
    input_variables=["old_screenshot"],
    template={"url": "data:image/png;base64,{old_screenshot}"}
)

# Action taken prompt
action_taken_prompt = PromptTemplate(
    input_variables=["action_taken"],
    template="{action_taken}"
)

reflection_human_prompt = HumanMessagePromptTemplate(prompt=[old_screenshot_prompt, action_taken_prompt, new_screenshot_prompt])

# Combined prompt
# TODO: Add the overall prompt message as well and say not to forget it (this should fix the 9am 9pm messup).
compare_screenshots_prompt = ChatPromptTemplate(
    input_variables=["new_screenshot", "action_taken", "input_annotated_screenshot", "input"],
    messages=[
        reflection_human_prompt,
        SystemMessagePromptTemplate.from_template("""
- Summarize the intention of the action.

- Summarize the result of the action by comparing the before (first) and after (second) screenshots. Be very careful \
here to check the details of the second screenshot. Don't make stuff up.

- State whether the result matched the intention to determine whether the action was a success or a failure.

- Note that the action may only be a partial step towards the "Thought" so if the action made progress towards the \
Thought, consider it a success. But if the action result did not match the action intent, consider the action a failure.

- If the thought is not fully completed by the action, also summarize the remaining steps required to complete the thought.

Importantly, do not forget your overall goal {input}
    """),
    ]
)