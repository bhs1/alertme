from alertme.utils.path_utils import get_logs_path, get_data_path


def get_html_content(page):
    return page.content()


def get_screenshot(page):
    return page.screenshot(full_page=True)


async def get_xpaths(page):
    return await page.evaluate(
        '''
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
    '''
    )


def write_html_content(content, action_name):
    logs_path = get_logs_path('html_logs')
    logfile_path = logs_path / f'{action_name.replace(" ", "_")}.txt'
    with open(logfile_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"HTML content saved to {logfile_path}")


def write_screenshot(screenshot, action_name):
    data_path = get_data_path('screenshots')
    screenshot_path = data_path / f'{action_name.replace(" ", "_")}.png'
    with open(screenshot_path, 'wb') as f:
        f.write(screenshot)
    print(f"Screenshot saved to {screenshot_path}")


def write_xpaths(xpaths, action_name):
    data_path = get_data_path('xpaths')
    xpath_logfile_path = data_path / f'{action_name.replace(" ", "_")}_xpaths.txt'
    with open(xpath_logfile_path, 'w', encoding='utf-8') as f:
        f.write("Textboxes:\n")
        for item in xpaths.get('textboxes', []):
            f.write(f"{item['xpath']} - {item['type']}\n")
        f.write("\nButtons:\n")
        for item in xpaths.get('buttons', []):
            f.write(f"{item['xpath']} - {item['type']}\n")
        f.write("\nRadio Buttons:\n")
        for item in xpaths.get('radios', []):
            f.write(f"{item['xpath']} - {item['type']}\n")
    print(f"XPaths logged to {xpath_logfile_path}")


def print_html_and_screenshot(page, action_name):
    html_content = get_html_content(page)
    screenshot = get_screenshot(page)
    write_html_content(html_content, action_name)
    write_screenshot(screenshot, action_name)


def extract_and_log_xpaths(page, action_name):
    xpaths = get_xpaths(page)
    write_xpaths(xpaths, action_name)
