from playwright.async_api import Page
import platform
from playwright_stealth import stealth_async

async def click(page: Page, x: float, y: float, text: str = "", exact: bool = False):
    if text:
        locator = page.get_by_text(text, exact=exact)
        if await locator.count() == 1:
            await locator.click()
            await page.wait_for_load_state('networkidle')
            return "locator"
        else:
            print(f"Multiple elements found for text: {text}")
            print(await locator.all_text_contents())
    
    await page.mouse.click(x, y)
    await page.wait_for_load_state('networkidle')
    return "coordinates"

async def type_text(page: Page, x: float, y: float, text: str):
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text)
    await page.keyboard.press("Enter")
    await page.wait_for_load_state('networkidle')

async def scroll(page: Page, x: float, y: float, direction: str, scroll_direction: int):
    if direction.upper() == "WINDOW":
        print(f"Scrolling window by {scroll_direction}")
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

async def go_back(page: Page):
    await page.go_back()

async def to_google(page: Page):
    await page.goto("https://www.google.com/")

async def to_url(page: Page, url: str):
    await page.goto(url)

async def display_red_dot(page, x, y):
    await page.evaluate("""
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
            setTimeout(() => dot.remove(), 1000);  // Remove dot after 10 seconds
        }
    """, {"x": x, "y": y})

from playwright.async_api import async_playwright

async def setup_browser():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True, args=None)
    page = await browser.new_page()
    await stealth_async(page)
    await page.set_viewport_size({"width": 1700, "height": 900})  # Set the viewport size
    await page.goto("https://www.google.com")
    
    
    return page