from datetime import datetime, timedelta
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
import argparse
import os
import re
import sys
import time
import cafe

# The [prev-month] year-month [next-month] row above the calendar:
#
#   <div class="layout row justify-space-between align-start py-3 px-4"><button type="button" class="v-btn v-btn--contained theme--light v-size--small primary"><span class="v-btn__content"><i aria-hidden="true" class="v-icon notranslate material-icons theme--light" style="font-size: 0.7rem;">chevron_left</i>
#                 12月
#               </span></button> <span class="body-1">2025年1月</span> <!----> <button type="button" class="v-btn v-btn--contained theme--light v-size--small primary"><span class="v-btn__content">
#                 2月
#                 <i aria-hidden="true" class="v-icon notranslate material-icons theme--light" style="font-size: 0.7rem;">chevron_right</i></span></button></div>

def prev_month_elem():
    """ Get prev-month element. """

    return WebDriverWait(driver, args.wait).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@type='button'][.//*[text()='chevron_left']]"))
    )

def year_month():
    """ Get year and month. """

    prev = prev_month_elem()
    text = prev.find_element(By.XPATH, "following-sibling::span").text
    match = re.search(r'(\d+)年(\d+)月', text)
    return [int(x) for x in match.group(1, 2)]

def next_month_elem():
    """ Get next-month element. """

    return WebDriverWait(driver, args.wait).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@type='button'][.//*[text()='chevron_right']]"))
    )

# The next-month element is disabled while the calendar is loading, so
# it can be used wait for the calendar to reload.
wait_for_calendar_reload = next_month_elem

def next_hour():
    """ Return next hour:  6:00, 7:00, 8:00 """

    hour = datetime.now() + timedelta(minutes=60)
    return hour.replace(minute=0, second=0, microsecond=0)

def restart_program():
    """ Restart the current program. """

    print("Restarting program...")
    driver.quit()
    os.execv(sys.executable, ['python'] + sys.argv)

def restart_if_error():
    # 10 minute timeout:
    #   10分経ってしまいました。はじめからお願いします。
    #   （ごめんなさい！)
    #
    # Congestion error:
    #   ごめんなさい！
    #   エラーが発生してしまいました。
    #
    # "Congestion" image link used for congestion and time out:
    #   https://kirbycafe-reserve.com/guest/_nuxt/img/3d4916f.png
    text = "ごめんなさい！"

    if text in driver.page_source:
        print("Congestion or 10 minute timeout!")
        restart_program()

def set_guests(num):
    """ Set number of guests. """

    # Click the "button" <div> since the <input> is not clickable.
    #
    #   <div role="button" aria-haspopup="listbox" aria-expanded="false" aria-owns="list-41" class="v-input__slot">
    #     <div class="v-select__slot">
    #       <div class="v-select__selections">
    #         <input id="input-41" readonly="readonly" type="text" aria-readonly="true">
    WebDriverWait(driver, args.wait).until(
        EC.presence_of_element_located((By.XPATH, "//div[@role='button']"))
    ).click()

    # The above click opens a listbox.  Getting DOM for listbox is not
    # obvious, so use keyboard instead of mouse.  Keyboard focus is
    # already on listbox.
    actions = ActionChains(driver) 

    # Just typing a number will make a selection and start loading
    # calendar.
    actions.send_keys(str(num))

    # Tab to collapse listbox.  Note that RETURN / ENTER will
    # sometimes leave the listbox open.
    actions.send_keys(Keys.TAB)  

    actions.perform()

# There's no info on any of the calendar seating.  So once we have a
# seat, we must calculate date/time based on position in table.
#
#   <table style>
#     <thead>...</thead
#     <tbody>
#       <tr>
#         <th>11:00</th>
#         <td class="">
#           <div>
#             <span>×</span>
#           </div>
#         </td>
#         <td class="">
#           <div>
#             <a>
#               <span>○</span>
#             </a>
#           </div>
#         </td>

def get_day(seat):
    """ Get day for seat by counting columns of <td> """

    tds = seat.find_elements(By.XPATH, "preceding-sibling::td")
    return 1 + len(tds)

def get_time(seat):
    """ Get time for seat from <th> at beginning of <tr> """

    return seat.find_element(By.XPATH, "preceding-sibling::th").text

def is_wanted_hour(seat):
    """ Return True if seat matches --hours """

    match = re.search(r'(\d+):\d+', get_time(seat))
    hour = int(match.group(1)) if match else -1
    return hour in want['hours']

# Map city to website.
city_site = {
    "tokyo":  "https://kirbycafe-reserve.com/guest/tokyo/",
    "osaka":  "https://osaka.kirbycafe-reserve.com/guest/osaka/",
    "hakata": "https://kirbycafe-reserve.com/guest/hakata/",
}

# Parse commandline arguments.
parser = argparse.ArgumentParser(
    prog='kirby-cafe',
    description='Kirby Cafe reservation')
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
    help='Month (1-12) for reservation date. To be used with --days. Defaults to latest month open for reservations.',
)
parser.add_argument(
    '-d', '--days',
    help='List / range of days for reservation date. Defaults to all days in month. For example, --days=1-5 is the first 5 days of --month.',
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
    help='Number of milliseconds it takes to get to server. Set --ping=-1 to ping the server.',
)
parser.add_argument(
    '-s', '--sleep',
    type=int,
    default=5,
    help='Number of seconds to sleep while reloading calendar. Defaults to "%(default)s"',
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
args = parser.parse_args()

# Set site based on city.
site = city_site[args.city]

# For parsed --hours and --days.
want = {}

# Parse --hours for list and ranges.
hours = args.hours
if hours:
    want['hours'] = cafe.list_and_ranges(hours)
    if not want['hours']:
        raise Exception(f"Bad --hours={hours}")

# Parse --days into xpath predicate.
days = args.days
if days:
    # Single day:  --days=7
    if re.match(r'^\d+$', days):
        want['days'] = days
    else:
        # List of days and day ranges:  --days=7,15-20,30
        if not re.match(r'^\d+(-\d+)?(,\d+(-\d+)?)*$', days):
            raise Exception(f"Bad --days {days}")
        want['days'] = " or ".join([
            "position()=" + x
            if x.isdecimal() else
            re.sub(r'^(\d+)-(\d+)$', r'(\1<=position() and position()<=\2)', x)
            for x in days.split(",")
        ])

# Get driver.
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=chrome_options)

# Starting at the top of the hour.
hour = next_hour()

# Number of milliseconds of ping to subtract.
ping = args.ping

# Ping the server.  
if ping == -1:
    ping = cafe.ping(site) / 2

if ping:
    print(f"Ping at {ping}ms")
    hour = hour - timedelta(milliseconds=ping)

print("Waiting for top of hour")
if not args.test:
    cafe.wait_until(hour)

if args.agree:
    # Start with the agreement page.
    print("Opening agreement page: " + site)
    driver.get(site)

    # Click on the label for the agree checkbox.  Clicking on the
    # agree checkbox itself does not work.
    elem = WebDriverWait(driver, args.wait).until(
        EC.presence_of_element_located((By.XPATH, "//label[@for='input-17']"))
    )
    print("Clicking agree")
    elem.click()

    # Go to reservation page.
    print("Clicking continue to reservation page")
    driver.find_element(By.PARTIAL_LINK_TEXT, "予約へ進む").click()
else:
    # Jump to reservation page.
    site += "reserve/"
    print("Opening reservation page: " + site)
    driver.get(site)

# Click "OK" on the 10 minute (10分...) pop-up warning.
elem = WebDriverWait(driver, args.wait).until(
    EC.presence_of_element_located((By.XPATH, "//span[text()='OK']"))
)
print("Clicking 'OK' to 10 minute time limit")
elem.click()

# Select number of guests.
print("Selecting number of guests: ", args.guests)
set_guests(args.guests)
wait_for_calendar_reload()

# Set month on calendar.  The calendar starts on the latest month open
# for reservations.  So if you're trying to get reservations for the
# next month on the 10th of this month, then --month is unnecessary.
month = year_month()[1]
if args.month:
    delta = args.month - month
    if delta != 0:
        # If previous month is December.
        if abs(delta) > 6:
            delta *= -1 
        if delta > 0:
            print("Clicking next month")
            next_month_elem().click()
        else:
            print("Clicking prev month")
            prev_month_elem().click()
        wait_for_calendar_reload()
        month = year_month()[1]

# Set xpath to get openings, which will be clickable with <a>.
if days:
    print(f"Searching all openings for {month}/{days}")
    xpath = f"//tr/td[{want['days']}][div/a]" 
else:
    print("Searching all openings for the month")
    xpath = "//td[div/a]"

# Get seat.
try:
    seat = cafe.get_seat(xpath, is_wanted_hour)
except NoSuchElementException:
    print("No seats available!")

    # Although there are no seats, if we wait, seats may become
    # available as people fail to complete reservation.
    i = 0
    while True:
        restart_if_error()

        if not i % 10:
            print("Reloading calendar...")
        i += 1

        if args.sleep:
            time.sleep(args.sleep)

        # Reload calendar by resetting number of guests.
        try:
            set_guests(args.guests)
            wait_for_calendar_reload()
        except TimeoutException:
            # Most likely, the calendar did not reload.  Possibly
            # because we're at an error page.
            print("Reload calendar failed...")
            restart_program()

        # Look for open seat.
        try:
            seat = driver.find_element(By.XPATH, xpath)
            break
        except NoSuchElementException:
            pass

# Get time based on position in table.
time = get_time(seat)

if days and days == want['days']:
    # Get day from --days=integer.
    day = days
else:
    # Get day based on position in table.
    day = get_day(seat)

# Click on the seat.  Note if someone else already clicked on this
# seat, it's already too late to try any other available seats.
print(f"Clicking [{month}/{day} at {time}]")
seat.click()

# Do not close.  Leave browser open to fill out form.
#   driver.quit()



