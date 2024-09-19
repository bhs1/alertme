from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

from alertme.utils.path_utils import get_output_path


async def display_red_dot(page, x, y):
    await page.evaluate(
        """
        ({x, y}) => {
            const dot = document.createElement('div');
            dot.style.position = 'fixed';
            dot.style.left = (x - 5) + 'px';  // Subtract half the width
            dot.style.top = (y - 5) + 'px';   // Subtract half the height
            dot.style.width = '10px';
            dot.style.height = '10px';
            dot.style.borderRadius = '50%';
            dot.style.backgroundColor = 'red';
            dot.style.zIndex = '2147483647';  // Maximum z-index value
            dot.style.pointerEvents = 'none'; // Ensure dot doesn't interfere with clicks
            document.body.appendChild(dot);
            setTimeout(() => dot.remove(), 10000);  // Remove dot after 10 seconds
        }
    """,
        {"x": x, "y": y},
    )


async def setup_browser():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    await stealth_async(page)
    await page.set_viewport_size({"width": 1700, "height": 900})  # Set the viewport size
    await page.goto("https://www.google.com")

    return page


async def save_page_html(page, filename='html_result.txt'):
    html_content = await page.content()
    output_file = get_output_path() / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Page HTML saved to {output_file}")
