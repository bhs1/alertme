import platform

from playwright.async_api import Page, expect

from alertme.web_voyager.playwright_utils import is_visible_in_viewport
from alertme.web_voyager.screenshot_utils import calculate_distance, mask_sensitive_data


async def wait_for_xpath(page, xpath, timeout=3000):
    try:
        locator = page.locator(f"xpath={xpath}")
        await locator.first.wait_for(state="visible", timeout=timeout)
    except Exception as e:
        print(f"Warning: XPath '{xpath}' not found within {timeout/1000} seconds: {e}")


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
            # print(f"visible_locators: {visible_locators}")
            clicked = False
            try:
                await visible_locators[0].click(timeout=500)
                clicked = True
            except Exception as e:
                print(f"Exception: {str(e)[:50]}...")
                print(f"Falling back to x,y since no locator found for text: {text}")
            await page.wait_for_load_state('networkidle')
            if clicked:
                print(f"Clicked text: {text}")
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
            print(
                f"Clicking locator with closest_locator distance: {locators_with_distance[0][1]} with text: {text}"
            )
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
    print(f"Typed text: {await mask_sensitive_data(text)}")


async def element_is_in_viewport(page: Page, locator, method="intersection_observer"):
    if method == "intersection_observer":
        try:
            print("Checking viewport...")
            # TODO(P2): Find a faster way to check if element is truly visible to user.
            await expect(locator).to_be_in_viewport(timeout=100)
            return True
        except Exception as e:
            print(f"Not visisble because: {str(e)[:50]}...")
            return False
    elif method == "javascript":
        return await is_visible_in_viewport(page, locator)


async def scroll_until_visible(
    page: Page, x: float, y: float, direction: str, scroll_direction: int, text: str = "", exact: bool = True
):
    max_scrolls = 3
    max_reverse_scrolls = 6
    max_final_scrolls = 6
    scroll_count = 0
    print(f"Scrolling until {text} is visible.")

    # Initial scroll
    for _ in range(max_scrolls):
        if await perform_scroll(page, x, y, direction, scroll_direction, text, exact):
            return True
        scroll_count += 1

    # Reverse scroll
    reverse_scroll_direction = -scroll_direction
    for _ in range(max_reverse_scrolls):
        if await perform_scroll(page, x, y, direction, reverse_scroll_direction, text, exact):
            return True
        scroll_count += 1

    # Final scroll to undo reverse scrolling
    for _ in range(max_final_scrolls):
        if await perform_scroll(page, x, y, direction, scroll_direction, text, exact):
            return True
        scroll_count += 1

    print(f"Element with text '{text}' not found after {scroll_count} total scrolls")
    return False


def is_window_scroll(x, y):
    return x == 0 and y == 0


async def perform_scroll(
    page: Page, x: float, y: float, direction: str, scroll_direction: int, text: str, exact: bool
):
    if text:
        try:
            visible_locators = await page.get_by_text(text, exact=exact).locator("visible=true").all()
            if len(visible_locators) > 0:
                for locator in visible_locators:
                    if await element_is_in_viewport(page, locator, method="intersection_observer"):
                        print(f"Found visible element with text: {text}")
                        return True
        except Exception as e:
            print(f"Exception while searching for text: {str(e)}")

    if is_window_scroll(x, y):
        print(f"Scrolling window by {scroll_direction}")
        current_scroll = await page.evaluate("window.pageYOffset")
        max_scroll = await page.evaluate("document.body.scrollHeight - window.innerHeight")

        if (scroll_direction > 0 and current_scroll >= max_scroll) or (
            scroll_direction < 0 and current_scroll <= 0
        ):
            print("Reached scroll limit. Moving to next phase.")
            return True

        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        print(f"Scrolling element at {x}, {y} by {scroll_direction}")
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    print(f"Waiting for 100ms")
    await page.wait_for_timeout(100)
    print("Done.")
    return False


async def go_back(page: Page):
    await page.go_back()


async def to_google(page: Page):
    await page.goto("https://www.google.com/")


async def to_url(page: Page, url: str):
    await page.goto(url)
