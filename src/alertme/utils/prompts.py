def get_reflection_prompt(last_test_results):
    return f"""(1) Determine what went wrong in each of the following test results: {last_test_results}.
Explain for each failing test case how does the actual output differ from the expected \
output (be very specific here, feel free to take multiple sentences)? And why do you think that is? \
If useful, consider the Debug Output when reasoning about your response.
    
    Note: To save resource, try to avoid printing entire HTTP requests or json, instead only summarize what may be useful for debugging \
    you may wish to summarize the structure in plain text the parts that you choose leave out.
     
    (2) Construct an improvement plan to fix the test failures. When formulating this plan, observe the previous solutions
    shown above and try not to repeat fixes that we already know do not lead to passing the test cases. Relfect explicitly on what
    we've tried before and whether it made a difference. If it did not make a difference, why not?
        
    (3) Sketch out the specific pseudocode of the fixes you plan to implement. Feel free to include actual code snipits.
    """


def get_reasoning_prompt(html, xpaths, url, prev_output, prev_debug, code_so_far):
    return f"""
            Current HTML:
            {html}
            Current XPaths:
            {xpaths}
            Current URL:
            {url}            
            
            Previous output: {prev_output}
            Previous debug output: {prev_debug}
            Previous code generated to get to this step: {code_so_far}
            
            Describe the current state of the page and the next navigation step we should take. Describe the html elements, xpaths, and UI
            components we should interact with to take the step.
            
            If we are already at the final page which can be used to achieve our final objective,
            add finish_code string to your output and say what code we should write to achieve the final objective. Otherwise, append
            continue_navigation string to your output, along with what you think we should do next to navigate closer to the final objective page.
            """


def get_generate_next_nav_step_prompt(reasoning_output):
    return f"""Consider the plan in {reasoning_output}. Write playwright code to \
proceed ONE more webpage step towards our final objective.
                """


def get_generate_prompt(
    reflection_prompt, reflection_summary, condensed_e2e_nav_code, condensed_e2e_nav_code_summary
):
    if not reflection_summary:
        reflection_summary = "No reflection summary yet since this is first iteration."

    return f"""We just gave you this prompt:\n\n {reflection_prompt} \n\n and as a result \
of the above prompt you gave this relfection summary:\n\n {reflection_summary}\n
Main directive: Rewrite the code to fix the test cases. Be sure to follow the plan \
in the reflection summary.

Here's the code that has been tested and gets us to the final page. You should use it to get to the final page
and then update the end of it to accomplish the main objective, given we've arrived at the final page.

Summary of code:
{condensed_e2e_nav_code_summary}

Code:
{condensed_e2e_nav_code}
             """


def get_generate_prompt_v2(reflection_prompt, reflection_summary):
    if not reflection_summary:
        reflection_summary = "No reflection summary yet since this is first iteration."

    return f"""We just gave you this prompt:\n\n {reflection_prompt} \n\n and as a result \
of the above prompt you gave this relfection summary:\n\n {reflection_summary}\n
Main directive: Rewrite the code to fix the test cases. Be sure to follow the plan \
in the reflection summary.
             """


def get_code_prompt_tuple(CODE_SUFFIX):
    prompt = f"""
    ===================== Coding agent general guideance =====================
- You are an expert programmer. Ensure any code you provide can be executed with all
required imports and variables defined. Structure your answer: 1) a prefix describing
any corrections, 2) the imports, 3) the functioning code block. The imports should only
contain import statements. E.g. import math, import sys, etc., not actual code.
Formatting the function you will write:
    - We will append the following code suffix to whatever code you generate:
{CODE_SUFFIX}         
    Don't generate the code suffix, we will append it for you after you generate main(str, page). But DO generate main(str, page)
    since it will be called in the suffix.
- Timeouts:
    - Keep timeouts to 3 seconds max. We have fast internet. Timeouts are often 30 seconds by default so if there is a possibility \
        to override the timeout, do so. Page.fill timeout should be 3 seconds.
- Exceptions
    - Don't raise uncaught exceptions. Instead, append to debug output and continue execution.
- Playwright: Use Playwright library not selenium. The "page" argument is a playwright page object. Be sure to use the \
playwright web driver and use the page object provided to you via the get_page() method above.
- Be sure to await any async methods. 
- Debugging:     
    - Print out intermediate variable names for debugging purposes. Instead of print() append
    to global variable debug_output. At first debug_output can be simple, but as you fail a test
    more times, you should add more and more details and variables to debug_output until we find
    the issue. Remember to define "global debug_output" at the top of any scope that updates debug_output.
    - It will be useful for you to print out HTTP requests and responses, HTML responses, and error logs in your debug output.
    These will be useful to help you debug your code. You should start doing this even in your first solution.
"""
    return ("user", prompt)
