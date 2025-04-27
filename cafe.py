# Common functions for pokemon-cafe.py and kirby-cafe.py

from datetime import datetime
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
import re
import subprocess
import time
import __main__

def get_seat(xpath, is_wanted_hour):
    """ Get an available seat """

    # Should explicitly pass these as arguments?
    driver = __main__.driver        
    args   = __main__.args          

    # How to filter available seats?
    index = args.index
    rand  = args.random
    hours = args.hours

    if index or rand or hours:
        # Get all available seats.
        print("Getting all seats")
        seats = driver.find_elements(By.XPATH, xpath)
        if not seats:
            raise NoSuchElementException
    else:
        # Get first available seat.  This is fastest, but if everyone
        # else is clicking on it, there's less chance of getting it.
        print("Getting first seat")
        return driver.find_element(By.XPATH, xpath)

    if hours:
        if index or rand:
            # Filter seats by hours.
            good = [s for s in seats if is_wanted_hour(s)]
            if good:
                print(f"Seats at --hours={hours}")
                seats = good
            else:
                print(f"Seats outside --hours={hours}")
        else:
            # Get first seat in --hours, or just first seat.
            for s in seats:
                if is_wanted_hour(s):
                    print(f"Getting first seat at --hours={hours}")
                    return s
            print(f"Getting first seat outside --hours={hours}")
            return seats[0]

    if index:
        print(f"Getting seat at --index={index}")
        return seats[index % len(seats)]
    if rand:
        print("Getting --random seat")
        return random.choice(seats)

def list_and_ranges(text):
    """ 
    Parse text for list and ranges.

    Lists are comma-separated, and ranges are dash-separated.  For example,
    text "7,15-20,30" should give set {7,15,16,17,18,19,20,30}.
    """

    if not text or not re.match(r'^\d+(-\d+)?(,\d+(-\d+)?)*$', text):
        return
    num = []
    for x in text.split(","):
        if x.isdecimal():
            num.append(int(x))
        else:
            start, stop = [int(n) for n in x.split("-")]
            for i in range(start, stop + 1):
                num.append(i)
    return set(num)

def ping(*sites):
    """ Ping sites.  Returns first successful ping. """

    for site in sites:
        try:
            server = re.sub(r'^\w+://', '', site, count=1)
            server = server.split("/")[0]
            print(f"Pinging: {server}")
            ping = subprocess.check_output(['ping', server]).decode('utf-8')
            ping = re.search(r'Average = (\d+)ms', ping).group(1)
            print(f"Ping ok: {ping}ms")
            return int(ping)
        except:
            print("Ping failed!")

    return 0

def wait_until(future):
    """ Wait until `future` datetime. """

    print("Wait until: ", future)
    while True:
        now = datetime.now()
        if now >= future:
            print("Wait over: ", now)
            break
        time.sleep((future - now).total_seconds() / 2)

