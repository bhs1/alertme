import os

def get_html_content(page):
    return page.content()

def get_screenshot(page):
    return page.screenshot(full_page=True)

async def get_xpaths(page):
    return await page.evaluate('''
        () => {
            const getXPath = (el) => {
                if (el.id) return `//*[@id="${el.id}"]`;
                const parts = [];
                while (el && el.nodeType === Node.ELEMENT_NODE) {
                    let idx = Array.from(el.parentNode.children).indexOf(el) + 1;
                    parts.unshift(`${el.tagName.toLowerCase()}[${idx}]`);
                    el = el.parentNode;
                }
                return `/${parts.join('/')}`;
            };

            const inputs = Array.from(document.querySelectorAll('input, textarea, select, button'));

            return {
                elements: inputs.map(el => ({
                    xpath: getXPath(el),
                    type: el.tagName.toLowerCase(),
                    inputType: el.type ? el.type : 'N/A'
                }))
            };
        }
    ''')

def write_html_content(content, action_name):
    os.makedirs('data/html_logs', exist_ok=True)
    html_log_path = f'data/html_logs/{action_name.replace(" ", "_")}.txt'
    with open(html_log_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"HTML content saved to {html_log_path}")

def write_screenshot(screenshot, action_name):
    os.makedirs('data/screenshots', exist_ok=True)
    screenshot_path = f'data/screenshots/{action_name.replace(" ", "_")}.png'
    with open(screenshot_path, 'wb') as f:
        f.write(screenshot)
    print(f"Screenshot saved to {screenshot_path}")

def write_xpaths(xpaths, action_name):
    os.makedirs('data/xpaths', exist_ok=True)
    xpath_log_path = f'data/xpaths/{action_name.replace(" ", "_")}_xpaths.txt'
    with open(xpath_log_path, 'w', encoding='utf-8') as f:
        f.write("Textboxes:\n")
        for item in xpaths.get('textboxes', []):
            f.write(f"{item['xpath']} - {item['type']}\n")
        f.write("\nButtons:\n")
        for item in xpaths.get('buttons', []):
            f.write(f"{item['xpath']} - {item['type']}\n")
        f.write("\nRadio Buttons:\n")
        for item in xpaths.get('radios', []):
            f.write(f"{item['xpath']} - {item['type']}\n")
    print(f"XPaths logged to {xpath_log_path}")

def print_html_and_screenshot(page, action_name):
    html_content = get_html_content(page)
    screenshot = get_screenshot(page)
    write_html_content(html_content, action_name)
    write_screenshot(screenshot, action_name)

def extract_and_log_xpaths(page, action_name):
    xpaths = get_xpaths(page)
    write_xpaths(xpaths, action_name)