from playwright.sync_api import sync_playwright
import os
from web_utils import print_html_and_screenshot, extract_and_log_xpaths

from playwright.sync_api import sync_playwright
import os
from web_utils import print_html_and_screenshot, extract_and_log_xpaths

def login_and_search_courts():
    print("Starting the browser...")
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)
    page = browser.new_page()

    try:
        print("Navigating to the login page...")
        page.goto('https://gtc.clubautomation.com/')
        page.wait_for_load_state('networkidle')
        print_html_and_screenshot(page, "navigation to login page")
        extract_and_log_xpaths(page, "navigation to login page")
        

        # Fill in login credentials from .env
        from dotenv import load_dotenv

        load_dotenv()

        username = 'bensc77'
        password = os.getenv('GGP_PASSWORD')

        print("Waiting for login form...")
        page.wait_for_selector('#login', state='visible')
        
        print("Filling in login credentials...")
        page.fill('#login', username)
        page.fill('#password', password)
        print_html_and_screenshot(page, "filling login credentials")
        extract_and_log_xpaths(page, "filling login credentials")
        
        print("Clicking login button...")
        page.click('#loginButton')
        print_html_and_screenshot(page, "clicking login button")

        print("Waiting for navigation to complete...")
        page.wait_for_load_state('networkidle', timeout=60000)  # 60 seconds

        print("Current URL after login:", page.url)

        print("Waiting for the 'Reserve a Court' button to be visible...")
        reserve_button_xpath = '//*[@id="menu_reserve_a_court"]'
        try:
            page.wait_for_selector(reserve_button_xpath, state='visible', timeout=30000)
            print("'Reserve a Court' button is now visible.")
            
            print("Clicking the 'Reserve a Court' button...")
            page.click(reserve_button_xpath)
            page.wait_for_load_state('networkidle', timeout=60000)
            print("Clicked 'Reserve a Court' button. New URL:", page.url)
            print_html_and_screenshot(page, "clicking Reserve a Court button")
            extract_and_log_xpaths(page, "clicking Reserve a Court button")

        except TimeoutError:
            print("'Reserve a Court' button not found or not visible within the timeout period.")

        print("Waiting for the page to load...")
        page.wait_for_load_state('networkidle', timeout=60000)  # 60 seconds

        print("Current page title:", page.title())
        print("Current URL:", page.url)

        print("Waiting for the search button to be visible...")
        search_button_selector = '#reserve-court-search'
        try:
            page.wait_for_selector(search_button_selector, state='visible', timeout=30000)
            print("Search button is now visible.")
            
            print("Setting the date for court search...")
            page.fill('#date', '07/03/2024')
            print_html_and_screenshot(page, "setting date for court search")
            extract_and_log_xpaths(page, "setting date for court search")

            print("Clicking the search button...")
            page.click(search_button_selector)
            print("Waiting for search results...")
            page.wait_for_selector('#times-to-reserve', timeout=30000)  # 30 seconds
            print_html_and_screenshot(page, "clicking search button")
            extract_and_log_xpaths(page, "clicking search button")

            print("Extracting available court times...")
            court_times = page.eval_on_selector_all('#times-to-reserve td', '''
                cells => cells.map(cell => {
                    const text = cell.textContent.trim();
                    const [location, ...times] = text.split('-').map(s => s.trim());
                    return {
                        location: location,
                        times: times[0].split(/\s+/).filter(t => t.includes('pm') || t.includes('am'))
                    };
                })
            ''')

            print("Available court times:")
            for court in court_times:
                print(f"  {court['location']}:")
                for time in court['times']:
                    print(f"    - {time}")
            print()  # Add an empty line for better readability

            # Take a screenshot of the search results
            os.makedirs('data', exist_ok=True)
            page.screenshot(path='data/court_times_screenshot.png', full_page=True)
            print("Screenshot saved to data/court_times_screenshot.png")

        except TimeoutError:
            print("Search button not found or not visible within the timeout period.")

    except Exception as e:
        print('An error occurred:', str(e))
        print_html_and_screenshot(page, "error occurred")
        extract_and_log_xpaths(page, "error occurred")                                    

    finally:
        print("Browser will remain open. Please close it manually when you're done.")
        input("Press Enter to close the browser and end the script...")
        browser.close()
        playwright.stop()

if __name__ == "__main__":
    print("Starting the script...")
    login_and_search_courts()
    print("Script finished.")
