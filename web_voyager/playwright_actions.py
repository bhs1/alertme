from playwright.async_api import Page
import platform
import asyncio

async def click(page: Page, x: float, y: float):
    await page.mouse.click(x, y)
    await page.wait_for_load_state('networkidle')

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

from playwright.async_api import async_playwright

async def setup_browser():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    await page.set_viewport_size({"width": 1700, "height": 900})  # Set the viewport size
    _ = await page.goto("https://www.google.com")
    return page