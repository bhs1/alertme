from playwright.async_api import Page
import platform
from playwright_stealth import stealth_async
from playwright_dompath.dompath_sync import xpath_path, css_path
import math

def calculate_distance(bbox, x, y):
    center_x = bbox['x'] + bbox['width'] / 2
    center_y = bbox['y'] + bbox['height'] / 2
    print(f"center_x: {center_x}, center_y: {center_y}")
    return math.sqrt((x - center_x)**2 + (y - center_y)**2)

async def click(page: Page, x: float, y: float, text: str = "", exact: bool = True):
    visible_locators = None
    if text:
        try:
            # Falling back to x,y since no locator found for text: Reserve a Court debug this.
            visible_locators = await page.get_by_text(text, exact=exact).locator("visible=true").all()
        except Exception as e:
            print(f"Exception: {str(e)[:50]}...")
            print(f"Falling back to x,y since no locator found for text: {text}")            
        if visible_locators is None or len(visible_locators) == 0:
            print(f"No visible elements found for text: {text}. Falling back to x,y coords.")
        elif len(visible_locators) == 1:
            print(f"Clicking locator using text search on text: {text}")
            #print(f"visible_locators: {visible_locators}")
            clicked = False
            try:
                await visible_locators[0].click(timeout=500)
                clicked = True
            except Exception as e:
                print(f"Exception: {str(e)[:50]}...")
                print(f"Falling back to x,y since no locator found for text: {text}")
            await page.wait_for_load_state('networkidle')
            if clicked:
                return "locator"
        elif len(visible_locators) > 1:
            print("========= Multiple locators found start. =========")
            print(f"x: {x}, y: {y}")
            
            locators_with_distance = []            
            for loc in visible_locators:
                bbox = await loc.bounding_box()
                print(bbox)
                distance = calculate_distance(bbox, x, y)
                print(distance)
                locators_with_distance.append((loc, distance))
            
            locators_with_distance.sort(key=lambda item: item[1])            
            closest_locator = locators_with_distance[0][0]
            print(f"Clicking locator with closest_locator distance: {locators_with_distance[0][1]} and text: {text}")
            clicked = False
            try:
                await closest_locator.click(timeout=500)
                clicked = True
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Falling back to x,y since no locator found for text: {text}")
            await page.wait_for_load_state('networkidle')
            if clicked:
                print("========= Multiple locators found end. =========")
                return "multiple_locators"
    
    print(f"Clicking x: {x}, y: {y}")
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

# First try to scroll into view using text argument and then fallback to coordinates.
async def scroll(page: Page, x: float, y: float, direction: str, scroll_direction: int, text: str = ""):
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
            setTimeout(() => dot.remove(), 10000);  // Remove dot after 10 seconds
        }
    """, {"x": x, "y": y})

from playwright.async_api import async_playwright

async def setup_browser():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    await stealth_async(page)
    await page.set_viewport_size({"width": 1700, "height": 900})  # Set the viewport size
    await page.goto("https://www.google.com")
    
    
    return page