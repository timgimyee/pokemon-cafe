from datetime import datetime, timedelta
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
import argparse
import cafe
import captcha
import os
import pickle
import re
import time

# The [prev-month] year-month [next-month] row above the calendar:
#
#   <div>
#   <a class="calendar-pager">前の月を見る<br>(Prev Month)</a>
#   </div>
#   <h3 style="font-size: 28px; letter-spacing: 4px; font-weight: 600;">
#   2024<span style="font-size: 14px; letter-spacing: 4px; font-weight: 600;">年</span>
#   12<span style="font-size: 14px; letter-spacing: 4px; font-weight: 600;">月</span>
#   </h3>
#   <div>
#   <a class="calendar-pager">次の月を見る<br>(Next Month)</a>
#   </div>

def prev_month_elem():
    """ Get prev-month element. """

    return WebDriverWait(driver, args.wait).until(
        EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, '(Prev Month)'))
    )

def year_month():
    """ Get year and month. """

    next = next_month_elem()
    text = next.find_element(By.XPATH, "../preceding-sibling::h3").text
    match = re.search(r'(\d+)年(\d+)月', text)
    return [int(x) for x in match.group(1, 2)]

def next_month_elem():
    """ Get next-month element. """

    return WebDriverWait(driver, args.wait).until(
        EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, '(Next Month)'))
    )

def next_step_button():
    """ Get (Next Step) button. """

    # The (Next Step) button:
    #
    #   <div class="button-container">
    #   <input type="submit" name="commit" value="次に進む (Next Step)" class="button" id="submit_button" data-disable-with="次に進む">
    #   </div>

    return WebDriverWait(driver, args.wait).until(
        EC.presence_of_element_located((By.XPATH, "//input[@id='submit_button' and contains(@value, '(Next Step)')]"))
    )

def next_twenty():
    """ 
    Return next 20 minute target time:  6:00, 6:20, 6:40, 7:00, 7:20

    Reservations start at the top of the hour.  Every 20 minutes
    afterwards, incomplete reservations are released.
    """

    dt = datetime.now() + timedelta(minutes=20)
    minute = dt.minute // 20 * 20
    return dt.replace(minute=minute, second=0, microsecond=0)

def refresh_while_congested():
    """ Refresh while at congested page. """

    text = "The site is congested due to heavy access"
    while True:
        if text in driver.page_source:
            print("Site congested!")

            # Do NOT click on blue Reload button!  It takes you back
            # to the start.  Refresh the browser instead.
            driver.refresh()
            time.sleep(args.sleep)
        else:
            print("No congestion!")
            break

def get_url(site):
    """ Get site and load cookies """

    driver.get(site)

    # Load cookies from previous visit.  This should allow us to skip
    # the security check puzzle if previously solved.
    #
    # Unfortunately, the more you use the cookies, the more the site
    # tracks you, and eventually, it will decide you are a bot.  Then
    # the tainted cookies will redirect you the the agreement page
    # when you click the (Next Step) button.
    if args.cookies:
        try:
            with open(cookies_file, 'rb') as file:
                for cookie in pickle.load(file):
                    driver.add_cookie(cookie)
            print("Loading cookies")
        except:
            print("No cookie file")
    else:
        # Silently remove cookie file.
        try:
            os.remove(cookies_file)
        except:
            pass

def security_check():
    """ Security check page.  Requires human to solve puzzle. """

    # Make sure page is fully loaded.
    WebDriverWait(driver, args.wait).until(
        lambda driver: driver.execute_script('return document.readyState') == 'complete'
    )
    page = driver.page_source

    # Did we skip security check?
    for text in ['(Make a Reservation)', 'Number of Guests']:
        if text in page:
            print(f"Skip security, text found: '{text}'")
            return

    # Solve captcha.
    if captcha.load_classifier():
        captcha.solve_captcha(driver)
    else:
        user_solve_captcha()

    # Save cookies so we can skip security check next time.
    if args.cookies:
        try:
            with open(cookies_file, 'wb') as file:
                pickle.dump(driver.get_cookies(), file)
            print("Saving cookies")
        except:
            print("Couldn't save cookies")

def user_solve_captcha():
    """ Wait for user to solve captcha. """

    #   <button class="amzn-captcha-verify-button btn btn-primary" id="amzn-captcha-verify-button" type="button" style="display: flex; padding: 5px 30px;">
    #   Begin<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iOSIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDkgMTQiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+IDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMS43ODQ2NyAwTDAuNTAwMjA1IDEuMjg0NDZMNi4yMTUwMyA3TDAuNTAwMjA1IDEyLjcxNTVMMS43ODQ2NyAxNEw4Ljc4NDY3IDdMMS43ODQ2NyAwWiIgZmlsbD0iYmxhY2siLz4gPC9zdmc+IA==" alt="begin" style="margin-left: 5px;">
    #   </button>

    # Click the "Begin >" button.
    WebDriverWait(driver, args.wait).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@id='amzn-captcha-verify-button']"))
    ).click()

    # Security check puzzle:
    #
    #   <canvas width="320" height="320">
    #   <button type="button" tabindex="0">1</button>
    #   <button type="button" tabindex="-1">2</button>
    #   <button type="button" tabindex="-1">3</button>
    #   <button type="button" tabindex="-1">4</button>
    #   <button type="button" tabindex="-1">5</button>
    #   <button type="button" tabindex="-1">6</button>
    #   <button type="button" tabindex="-1">7</button>
    #   <button type="button" tabindex="-1">8</button>
    #   <button type="button" tabindex="-1">9</button>
    #   </canvas>
    #   ...
    #   <button type="submit" id="amzn-btn-verify-internal" class="btn btn-primary" style="display: block;">Confirm</button>
    #   ...
    #   <select class="amzn-captcha-lang-selector" aria-label="Select language">

    # Use the language selector instead of "Confirm" button.  If you
    # fail the puzzle, the "Confirm" button is temporarily replaced
    # with a "Loading" button for new puzzle.
    lang = "//select[@class='amzn-captcha-lang-selector']"

    # Wait for puzzle to load.
    WebDriverWait(driver, args.wait).until(
        EC.presence_of_element_located((By.XPATH, lang))
    )

    # Wait for user to solve puzzle.  I needed more than args.wait=10
    # seconds :/
    print("Please solve the puzzle")
    WebDriverWait(driver, 600).until(
        EC.invisibility_of_element_located((By.XPATH, lang))
    )
    print("Puzzle solved!")

def set_month(target_month):
    """ Set month on calendar. """

    # Get the current displayed date.
    year, month = year_month()

    # Keep clicking (Next Month) to get to target_month.
    while month != target_month:
        next_month_elem().click()

        # Wait for displayed date to change.
        old_month = month
        while month == old_month:
            year, month = year_month()

def get_open_day():
    """ Get open day for --days. """

    while True:
        # When a date is "Full", "N/A", or grayed out:
        #
        #   <li class="calendar-day-cell not-available" ...>

        # Get all open dates.
        dates = driver.find_elements(By.XPATH, f"//li[contains(@class, 'calendar-day-cell') and not(contains(@class, 'not-available'))]")

        # Get first date that matches --days.
        for date in dates:
            day = int(date.text.split()[0])
            if day in want['days']:
                return day

        # Wait a bit before checking again.
        time.sleep(args.slumber)

        # Reload the calendar by going to (Next Month) then returning
        # to (Prev Month).
        next_month_elem().click()
        prev_month_elem().click()

def is_wanted_hour(seat):
    """ Return True if seat matches --hours """

    match = re.search(r'(\d+):\d+~', seat.text)
    hour = int(match.group(1)) if match else -1
    return hour in want['hours']

# Map city to website.
city_site = {
    "tokyo":  "https://reserve.pokemon-cafe.jp/",
    "osaka":  "https://osaka.pokemon-cafe.jp/",
}

# Parse commandline arguments.
parser = argparse.ArgumentParser(
    prog='pokemon-cafe',
    description='Pokemon Cafe reservation')
parser.add_argument(
    '-c', '--city',
    choices=list(city_site),
    default='tokyo',
    help='Select city. Defaults to "%(default)s"',
)
parser.add_argument(
    '-g', '--guests',
    type=int,
    default=2,
    help='Number of guests (1-6). Defaults to %(default)s.',
)
parser.add_argument(
    '-m', '--month',
    type=int,
    help='Month (1-12) for reservation date. To be used with --days. Defaults to current month.',
)
parser.add_argument(
    '-d', '--days',
    help='List / range of days for reservation date. Defaults to 31 days from now. For example, --days=1-5 is the first 5 days of --month.',
)
parser.add_argument(
    '-H', '--hours',
    help='List / range of preferred hours. For example, --hours=16-24 is 4pm and later.',
)
parser.add_argument(
    '-i', '--index',
    type=int,
    default=0,
    help='If multiple available seats, get seat by index. Defaults to first seat. For example, --index=-1 is the last seat.',
)
parser.add_argument(
    '-r', '--random',
    action='store_true',
    help='If multiple available seats, get seat randomly. Defaults to first seat.',
)
parser.add_argument(
    '-a', '--agree',
    action='store_true',
    help='Go through the agreement page.  Default is to jump directly to reservation page.',
)
parser.add_argument(
    '-p', '--ping',
    type=int,
    default=0,
    help='Number of milliseconds it takes to get to server.',
)
parser.add_argument(
    '-s', '--sleep',
    type=int,
    default=1,
    help='Number of seconds to sleep while refreshing a congestion page. Defaults to "%(default)s"',
)
parser.add_argument(
    '-S', '--slumber',
    type=int,
    default=30,
    help='Number of seconds to sleep while checking specific --month and --days. Defaults to "%(default)s"',
)
parser.add_argument(
    '-w', '--wait',
    type=int,
    default=60,
    help='Number of seconds to wait for page or page elements to load or change. Defaults to "%(default)s"',
)
parser.add_argument(
    '-T', '--test',
    action='store_true',
    help='For testing.',
)
parser.add_argument(
    '-C', '--cookies',
    action='store_true',
    help='Accept cookies.  Remembers if security check puzzle was previously solved.',
)
args = parser.parse_args()

# File to store cookies.
cookies_file = 'pokemon-cookies.pkl'

# Set website url based on city.
site = city_site[args.city]

# Parse --days and --hours for list and ranges.
want = {x: getattr(args, x) for x in ['days', 'hours']}
for key, val in want.items():
    if val:
        want[key] = cafe.list_and_ranges(val)
        if not want[key]:
            raise Exception(f"Bad --{key}={val}")

# Pre-load classifier before opening browser.
captcha.load_classifier()

# Get driver.
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=chrome_options)

if args.agree:
    # Start with the agreement page.
    print("Opening agreement page: " + site)
    get_url(site)

    # Click on the label for the agree checkbox.  Clicking on the
    # agree checkbox itself does not work.
    #
    #   <div class="button-container">
    #   <input type="checkbox" id="agreeChecked" name="agreeChecked" value="true">
    #   <label for="agreeChecked" class="agreeChecked">同意する / Agree to terms</label>
    #   </div>
    WebDriverWait(driver, args.wait).until(
        EC.element_to_be_clickable((By.XPATH, "//label[@class='agreeChecked']"))
    ).click()

    # Click "Go to the Reservation Page" button.
    #
    #   <div class="button-container-agree">
    #   <button class="button">同意して進む<br>（Go to the Reservation Page ）</button>
    #   </div>
    WebDriverWait(driver, args.wait).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@class='button']"))
    ).click()

    # Human?
    security_check()

    # Go to reservation page.
    #
    #   <a class="button arrow-down" href="/reserve/step1">予約へ進む<br>(Make a Reservation)</a>
    WebDriverWait(driver, args.wait).until(
        EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, '(Make a Reservation)'))
    ).click()

    # Sometimes there's a 2nd security check even though we just passed the 1st.
    security_check()
else:
    # Jump to reservation page.
    site += "reserve/step1"
    print("Opening reservation page: " + site)
    get_url(site)

    # Human?
    security_check()

# Select number of guests.
#
#   <select onchange="this.form.submit()" name="guest"><option value="0">選択してください</option>
#   <option value="1">1名</option>
#   <option value="2">2名</option>
#   <option value="3">3名</option>
#   <option value="4">4名</option>
#   <option value="5">5名</option>
#   <option value="6">6名</option></select>
elem = WebDriverWait(driver, args.wait).until(
    EC.presence_of_element_located((By.NAME, 'guest'))
)
guests = str(args.guests)
print("Setting number of guests: " + guests)
Select(elem).select_by_value(guests)

# Wait for calendar and (Next Step) button to load.
next_step_button()

# Set target_date either 31 days from now or an open date in --days.
target_day = 0

if want['days']:
    # Wait for one of --days to open.

    if args.month:
        set_month(args.month)
    target_day = get_open_day()

    year, month = year_month()
    dt = datetime(year, month, target_day, 0, 0, 0)
    print("Found open date at", dt.strftime("%B %d"))
else:
    # Calculate 31 days from now.

    num = 30 if args.test else 31
    dt = datetime.now() + timedelta(days=num)
    print(f"{num} days from now is {dt.strftime('%B %d')}")

    set_month(dt.month)
    target_day = dt.day

# Click target_date on the calendar.
#
# The calendar will include dates from previous and next months grayed
# out using "opacity: 0.3" style.  Even though grayed out, these dates
# are still clickable.
#
# If 31 days from now, the date will be N/A but still clickable.
driver.find_element(By.XPATH, f"//li[contains(@class, 'calendar-day-cell')]/div[contains(text(), '{target_day}') and not(contains(@style, 'opacity'))]").click()

# Get (Next Step) button again, in case clicking the calendar changes
# anything.
next_step = next_step_button()

# Wait until 6:00, 6:20, 6:40, 7:00, etc.  This program should be run
# 5-10 minutes before.  Too early runs the risk of session timeout.
# Too late runs the risk of traffic congestion.
if not want['days']:
    dt = next_twenty()
    
    # Number of milliseconds of ping to subtract.
    ping = args.ping

    # Ping the server.  Unfortunately, the Pokemon Cafe server is
    # unpingable.  Hopefully, pinging Kirby Cafe is close enough.
    if ping == -1:
        ping = cafe.ping(site, "http://kirbycafe-reserve.com") / 2

    if ping:
        print(f"Ping at {ping}ms")
        dt = dt - timedelta(milliseconds=ping)
    
    print("Waiting for next 20 minute interval")
    if not args.test:
        cafe.wait_until(dt)

# Click (Next Step) button.
print("Clicking (Next Step) button")
next_step.click()

# The site will be congested.
refresh_while_congested()

# Page with time slots has the following text, but it fails to match:
#
#    Please select the day and time that you would like your 
#    table reservation
#
# Page also has this text, which does match:
# 
#    Please note that your reservation will only be available
#    for 90 minutes regardless of your arrival time

good_text = "90 minutes"

# Tainted cookies or clicking (Next Step) too early will redirect to
# the agreement page.

bad_text = "Agree to terms"

while True:
    if good_text in driver.page_source:
        print(f"Text found: '{good_text}'")
        break
    elif bad_text in driver.page_source:
        print(f"Redirected to start.  Text found: '{bad_text}'")
        try:
            os.remove(cookies_file)
        except:
            pass
        exit(1)
    else:
        print(f"Text not found: '{good_text}' ...refreshing page.")
        driver.refresh()
        time.sleep(args.sleep)

# An "Available" seat:
#
#   <td>
#     <div class="time-cell">
#       <a class="level post-link" data-seat-id="163490" data-guest="2" href="javascript:void(0)">
#         <div class="seattypetext level-left">C席</div>
#         <div class="timetext level-left">13:00~</div>
#         <div class="status-box">
#           <div class="status level-left">空席</div>
#           <div class="status level-left">Available</div>
#         </div>
#       </a>
#     </div>
#   </td>
#
# A "Full" seat:
#
#   <td>
#     <div class="time-cell">
#       <div class="level full">
#         <div class="seattypetext level-left">A席</div>
#         <div class="timetext level-left">18:00~</div>
#         <div class="status-box">
#           <div class="status level-left">満席</div>
#           <div class="status level-left">Full</div>
#         </div>
#       </div>
#     </div>
#   </td>

# Path to available seats.
if args.test:
    xpath = f"//td/div[@class='time-cell']/div[@class='level full']"
else:
    xpath = f"//td/div[@class='time-cell']/a"

# Get seat.
try:
    seat = cafe.get_seat(xpath, is_wanted_hour)
except NoSuchElementException:
    print("No seats available!")
    if not args.test:
        driver.quit()
    exit(0)

# Get reservation text (seat and time) before clicking.
text = seat.text

# Click on the seat.  Note if someone else already clicked on this
# seat, it's already too late to try any other available seats.
seat.click()

# Remove Japanese text and newlines from reservation text.
text = re.sub(r'(?:\s|[^\x00-\x7F])+', ' ', text)
print(f"Clicked: [{text}]")

# Refresh until we get to form.
refresh_while_congested()

# Do not close.  Leave browser open to fill out form.
#   driver.quit()




