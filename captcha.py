# This module will solve the Pokemon Cafe captcha.  But before that:
#
#     "Let's confirm you are human"
#
# The first step is to collect images from the captcha:
#
#     python -u captcha.py --collect-images
#
# This will take you to a modified captcha.  Instead of a "Confirm"
# button, there will be a "Save Images" button.
#
# Solve the captcha as usual by clicking on the correct images.  There
# will be 5 correct images, but you can click on fewer if you cannot
# find 5 with certainty.
#
# Click on the "Save Images" button to save the images.  There is no
# check that the images are correct, so be sure to check the
# correctness yourself.
#
# After saving, a new captcha will load.  Solve and click "Save
# Images" again.  Repeat.
#
# Repeat until you have at least 50 images per category.  The images
# are used to build the classifier, so more is better.
#
# Next, build the classifier.  Run with --test to get an accuracy and
# classification report:
#
#     python -u captcha.py --build-classifier --test
#
# If the accuracy is low, go collect more images.  Once you are happy
# with the accuracy, run without --test to build a final classifier:
#
#     python -u captcha.py --build-classifier
#
# The Pokemon Cafe script should now automatically solve the captcha:
#
#     python -u pokemon-cafe.py

from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from skimage.metrics import structural_similarity as ssim
import argparse
import cv2
import joblib
import numpy as np
import os
import time
import tkinter

# These modules are imported in functions below:
#
#   from sklearn import svm
#   from sklearn.metrics import accuracy_score
#   from sklearn.metrics import classification_report
#   from sklearn.model_selection import train_test_split
#   from sklearn.model_selection import GridSearchCV

# Directory for saving images.
IMAGE_DIR = 'images'

# The captcha canvas is 320x320 divided into a 3x3 grid of 104x104
# images separated by 4px of whitespace.  Clicking on the whitespace
# is the same as clicking on the image to its left or its top; in
# other words, the clickable area of each image is actually 108x108,
# and the top-left corner of each image will be at multiples of 108.
#
# Note: Although the captcha seems to support older browsers with
# alternative content <button>s, this script will only work with the
# <canvas> itself:
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

IMAGE_SIZE = 104
CLICK_SIZE = 108

# Crop dimensions for each image position.
IMAGE_CROP = {}
for j in range(3):
    for i in range(3):
        l = i * CLICK_SIZE
        t = j * CLICK_SIZE
        r = l + IMAGE_SIZE
        b = t + IMAGE_SIZE
        IMAGE_CROP[f"{i}{j}"] = [l, t, r, b]

# File to store image classifier model.
MODEL_FILE = 'model.joblib'

# File to store best params for creating model.
BEST_PARAMS_FILE = 'best_params.joblib'

# Threshold at which two images are probably duplicates.  Duplicate
# images will have a structural similarity score closer to 1.  From
# testing, almost all duplicates scored above 0.85 with only one
# scoring below at 0.82; almost all non-duplicates scored below 0.75
# with only two scoring above at 0.76 and 0.79.
SSIM_THRESHOLD = 0.81

#### Main Functions ####

def collect_images(set_driver=None, url=None, ssim_threshold=None, test=False):
    """
    Collect images from captcha.

    The "set_driver" parameter will set the global web driver.

    The "url" parameter will open the web page for the captcha.

    The "ssim_threshold" parameter determines if two images are
    duplicates.

    The "test" parameter will double-check puzzle clicks.
    """

    # Set web driver.
    driver = web_driver(set_driver)

    # Open browser to captcha.
    if url:
        driver.get(url)

    # Default ssim_threshold here in case None is explicitly passed in.
    if ssim_threshold is None:
        ssim_threshold = SSIM_THRESHOLD

    # Navigate captcha intro.
    begin_page()

    # Add "Save Images", "Saving" and "Quit" buttons.
    add_buttons()

    while True:
        # Reset listeners and buttons.
        initiate_puzzle()

        # Get instruction category, which tells which images to
        # select.
        cat = instruction_cat()
        print(f"Found category from instruction: {cat}")

        # Make directory for category.
        path = IMAGE_DIR + "/" + cat
        os.makedirs(path, exist_ok=True)

        # Get screenshot of canvas.
        screenshot = screenshot_canvas()

        # Crop out 9 puzzle images.
        tile = {k: screenshot[t:b, l:r] for k, (l,t,r,b) in IMAGE_CROP.items()}

        # Wait for user to click "Save Images" or "Quit" button.
        WebDriverWait(driver, 600).until(
            EC.invisibility_of_element_located((By.XPATH, "//button[@id='zzz-save']"))
        )

        # Get the clicked image coordinates.
        canvas = canvas_element()
        mimikyu = canvas.get_attribute('data-mimikyu')

        # User clicked "Save Images" without clicking puzzle.
        if not mimikyu:
            print("No clicked images.  Getting new puzzle.")
            get_new_puzzle()
            continue

        # User clicked "Quit" button.
        if mimikyu == "QUIT":
            print("Stop collecting images!")
            driver.quit()
            return

        print("User clicked on: ", mimikyu)

        # Clicking twice unclicks, so filter for odd numbered clicks. 
        coords = [ij for ij, c in Counter(mimikyu.split()).items() if c % 2 == 1]
        print("True clicks: ", coords)

        # User clicked "Save Images" without selecting any images.
        # All puzzle clicks were unclicked.
        if not coords:
            print("No selected images.  Getting new puzzle.")
            get_new_puzzle()
            continue

        # Get the canvas location and size.  The idea is to create an
        # image viewer in the same place with the same dimensions, so
        # when the browser minimizes, the image viewer is right where
        # the user was just looking.
        x, y, w, h = canvas_area()

        # Remove image viewer's titlebar height, so image viewer
        # screen coincides with canvas area.  A Tk window is
        # temporarily created under the browser (specifically under
        # the canvas area), so it is never seen by the user.
        y = y - titlebar_height(x, y, w, h)

        # Double-check that correct images were clicked.
        if test:
            # Title for image viewer.
            title = f"Are these all {cat}? [y/n]"

            # Create window for image viewer.  Like Tk windows, the
            # image viewer is created in a layer below the browser.
            cv2.namedWindow(title)

            # Place image viewer under puzzle canvas.
            cv2.moveWindow(title, x, y)

            # The original intent was to bring the image viewer to the
            # top without minimizing the browser, but the browser held
            # on to the focus.  And minimizing the browser afterwards
            # sent the focus to the command line instead of the image
            # viewer.
            #
            # cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)

            # Minimize browser window to relinquish focus.  Focus
            # should now go to image viewer.
            driver.minimize_window()

            # Filter out mistaken clicks.
            ok_coords = []

            for ij in coords:
                # Show selected image.
                cv2.imshow(title, tile[ij])

                # Resize window to puzzle canvas size.  Otherwise,
                # image viewer will default to image size, and images
                # are tiny and easy to miss.
                cv2.resizeWindow(title, w, h)

                # Press "N" or "n" to unselect image.
                if chr(cv2.waitKey(0)) not in 'Nn':
                    ok_coords.append(ij)

            # Destroy window.
            cv2.destroyWindow(title)

            coords = ok_coords
            print("Double-checked clicks: ", coords)

        # Create the base part of a unique file name.
        base = path + "/" + datetime.now().strftime('%Y%m%d_%H%M%S')

        # Get pre-existing image files.
        files = glob(f"{IMAGE_DIR}/{cat}/*.png")

        # Save images to file.
        for ij in coords:
            img = tile[ij]

            # Check for duplicates.
            if is_dup_image(img, files, ssim_threshold, x, y, w, h):
                continue

            # Write image.
            file = f"{base}-{ij}.png"
            print(f"Writing image to: {file}")
            cv2.imwrite(file, img)

            # Add image to existing files.
            files.append(file)

        # Show number of images collected so far.
        show_image_count()

        # Get new puzzle.  This will automatically restore the
        # minimized browser.
        get_new_puzzle()

# Resources for image classifications using Support Vector Machines:
#
#   https://www.geeksforgeeks.org/image-classification-using-support-vector-machine-svm-in-python/
#   https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#
# Here is an example run of build_classifier() with about 1500 images:
#
#   Loading category: 348 bags
#   Loading category: 385 beds
#   Loading category: 209 buckets
#   Loading category: 124 chairs
#   Loading category: 206 clocks
#   Loading category: 80 curtains
#   Loading category: 175 hats
#   Loaded best_params.joblib: {'C': 0.1, 'gamma': 0.0001, 'kernel': 'poly'}
#   Start training model: 2025-06-14 03:03:30.166440
#   Finish training model, elapsed time: 0:06:33.726592
#   The model is 80.1% accurate
#                 precision    recall  f1-score   support
#   
#           bags       0.75      0.83      0.79        70
#           beds       0.88      0.87      0.88        77
#        buckets       0.78      0.90      0.84        42
#         chairs       0.58      0.60      0.59        25
#         clocks       0.92      0.85      0.89        41
#       curtains       0.90      0.56      0.69        16
#           hats       0.77      0.66      0.71        35
#   
#       accuracy                           0.80       306
#      macro avg       0.80      0.75      0.77       306
#   weighted avg       0.81      0.80      0.80       306

def build_classifier(best_params=None, slicer=None, model_file=None, test=False):
    """
    Build an SVC model for classifying images.

    The "best_params" parameter is a dictionary with parameters for
    svm.SVC().  For example:

        best_params = {'C': 0.1, 'gamma': 0.0001, 'kernel': 'poly'}

    If None, will use parameters from BEST_PARAMS_FILE or
    grid_search_best_params().

    The "slicer" parameter is a slice() object to be applied to image
    files in each category.

    The "model_file" parameter is where the model is saved.  Defaults
    to MODEL_FILE.

    The "test" parameter will test the model after training.
    """

    # Import modules here since they take a while to load.
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    # Slice image files in each category.
    cat_files = slice_category_files(slicer)

    # Load images into numerical arrays.
    X, y = load_image_data(cat_files)

    if test:
        # Set some data aside for testing.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y)
    else:
        # Train with all data.
        X_train, y_train = X, y

    # Get best parameters.
    if not best_params:
        if os.path.exists(BEST_PARAMS_FILE):
            best_params = joblib.load(BEST_PARAMS_FILE)
            print(f"Loaded {BEST_PARAMS_FILE}: {best_params}")
        else:
            best_params = grid_search_best_params(max_images=5)
            print(f"grid_search_best_params: {best_params}")

    # Create SVC model.
    model = svm.SVC(probability=True, **best_params)

    # Fit data to model.
    start_dt = datetime.now()
    print("Start training model:", start_dt)
    model.fit(X_train, y_train)
    print("Finish training model, elapsed time:", datetime.now() - start_dt)

    # Save model to file.
    joblib.dump(model, model_file or MODEL_FILE)

    # Print accuracy and classification report.
    if test:
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        print(f"The model is {accuracy:.1%} accurate")

        categories = sorted(cat_files.keys())
        print(classification_report(y_test, y_pred, target_names=categories))

    return model

def solve_captcha(set_driver=None, url=None, test=False):
    """
    Solve the captcha.

    The "set_driver" parameter will set the global web driver.

    The "url" parameter will open the web page for the captcha.

    The "test" parameter will wait for user input after solving, so
    user has time to double-check.
    """

    # Get the classifier.  Ideally, this was called earlier and
    # already in-memory.
    classifier = get_classifier()

    # Set web driver.
    driver = web_driver(set_driver)

    # Open browser to captcha.
    if url:
        driver.get(url)

    # Category names and index.
    categories = sorted( d.name for d in os.scandir(IMAGE_DIR) if d.is_dir() )
    cat_index = {cat: i for i, cat in enumerate(categories)}

    # Navigate captcha intro.
    begin_page()    

    while True:
        # Get instruction category, which tells which images to
        # select.
        cat = instruction_cat()
        idx = cat_index[cat]
        print(f"Found category from instruction: {cat}")

        # Get screenshot of canvas.
        screenshot = screenshot_canvas()

        # Data for each image in dictionary with named keys.  To be
        # displayed by show_puzzle_solution().
        data = {}

        # Data for each image in flat list for easy sorting.
        tiles = []

        # Number of images matching instruction category.
        matches = 0

        # Loop through each image.
        for ij, (l, t, r, b) in IMAGE_CROP.items():
            # Crop out image and flatten.
            crop = screenshot[t:b, l:r] 
            flat = [crop.flatten()]

            # Predicted category for this image.
            pred = classifier.predict(flat)[0]

            # Probability model for this image.  This uses cross
            # validation, so can be slightly different from predict().
            pp = classifier.predict_proba(flat)[0]

            # Probability that this image belongs to the instruction
            # category.
            prob = pp[idx]

            # Does predicted category match the instruction category?
            # Use integers 1 and 0 because they are sortable and
            # truthy.
            match = 1 if pred == idx else 0

            if match:
                matches += 1

            # Data for show_puzzle_solution().
            data[ij] = {
                "match": match,
                "cat":   categories[pred],
                "prob":  prob,
            }

            # Data in sortable order.
            tiles.append([match, prob, categories[pred], ij])

        # Sort tiles by matches then by probability.
        tiles.sort(reverse=True)

        # Report number of matches.
        if matches == 5:
            print(f"Found exactly 5 {cat} in puzzle.")
        elif matches >= 5:
            print(f"Found {matches} {cat} in puzzle.  Using 5 highest probabilities.")

            # Click on the extra matches now, so they unclick when all
            # the matches are clicked later.
            for t in tiles[5:matches]:
                ij = t[-1]
                click_tile(ij)
        elif matches == 4:
            print(f"Found only 4 {cat} in puzzle.")

            # Auto-promote the 5th image.  There is no penalty for
            # sending an incorrect solution.
            match, prob, pred_cat, ij = tiles[4]
            matches = 5
            data[ij][0] = 1

            print(f'Promoting "{pred_cat}" image to "{cat}" with {prob:.1%}')
        elif matches:
            print(f"Found only {matches} {cat} in puzzle.")
        else:
            print(f"Found no {cat} in puzzle.")

        show_puzzle_solution(cat, data)

        # Click on the matches.
        for t in tiles[0:matches]:
            ij = t[-1]
            click_tile(ij)

        # Wait a bit, so there's time to check puzzle solution.
        if test:
            if input("Continue? [y/n]").lower() == 'n':
                break
        else:
            pause(3, 5)

        # Puzzle failed.
        if matches < 5:
            print("Getting new puzzle.")
            get_new_puzzle()
            continue

        # Puzzle solved!  Maybe.  Click on the "Confirm" button to see.
        id = confirm_button_id()
        driver.find_element(By.XPATH, f"//button[@id='{id}']").click()

        # Clicking the "Confirm" button may give an alert:
        #
        #   <div role="alert" tabindex="-1" style="margin-bottom: 20px; display: none; background-color: rgb(255, 233, 228); padding: 5px 20px; font-size: 0.875em; outline: red solid 2px;">
        #   <img src="data:image/svg+xml;base64,...+IA==" role="presentation" alt="">
        #   <p style="margin-left: 10px;">Time limit exceeded. Please try again.</p>
        #   </div>

        alert_xpath = "//div[@role='alert']/p"

        # Or it may solve the puzzle and leave the captcha page, in
        # which case, the captcha's language selector disappears:
        #
        #   <select class="amzn-captcha-lang-selector" aria-label="Select language">

        lang_xpath = "//select[@class='amzn-captcha-lang-selector']"

        try:
            status = WebDriverWait(driver, 20).until(
                EC.any_of(
                    EC.visibility_of_element_located((By.XPATH, alert_xpath)),
                    EC.invisibility_of_element_located((By.XPATH, lang_xpath))
                )
            )
        except TimeoutException:
            # No alert message and still on captcha page.  Don't know
            # what to do, so punt.
            print("TimeoutException.  No alert.  Unsolved captcha.")
            reload_captcha()
            continue

        # Left the captcha page.  Puzzle solved!  Yay~!
        if status is True:
            print("Puzzle solved!")
            return

        alert = status.text
        print(f"Alert: {alert}")

        # Alert for wrong answer:
        #
        #   "Incorrect. Please try again."

        if alert.startswith("Incorrect"):
            # The alert box remains while a new puzzle is loading, with
            # the "Loading" button replacing the "Confirm" button.
            WebDriverWait(driver, 100).until(
                EC.invisibility_of_element_located((By.XPATH, alert_xpath))
            )
            wait_for_puzzle()
            continue

        # Alert for taking too long with entire captcha:
        #
        #   "Time limit exceeded. Please refresh the page."

        if alert.startswith("Time limit"):
            reload_captcha()
            continue

        # Alert for taking too long with a single puzzle:
        #
        #   "Time limit exceeded. Please try again."
        #
        # This comment is just bookkeeping.  There is no check for
        # this alert, which only happens before clicking the "Confirm"
        # button.

        # Unknown alert.  Punt.
        print("Unknown alert:", alert)
        reload_captcha()

def dedup_images(category=None, ssim_threshold=None, x=None, y=None, w=350, h=200):
    """
    De-duplicate images, removimg images that appear multiple times in
    a category.

    The "category" parameter is the category to check.  The default is
    to check all categories.

    The "ssim_threshold" parameter determines if images are the same.
    The default is a bit less than SSIM_THRESHOLD.

    The "x", "y", "w" and "h" parameters are for image viewer
    placement.
    """

    # The default SSIM_THRESHOLD used in collect_images() may miss
    # some duplicate images, so lower the threshold a bit to catch
    # them.
    if ssim_threshold is None:
        ssim_threshold = SSIM_THRESHOLD - 0.05

    # Center horizontally.
    if x is None:
        screen_w, screen_h = screen_size()
        x = int((screen_w - w) / 2)

    # Center vertically.
    if y is None:
        screen_w, screen_h = screen_size()
        y = int((screen_h - h) / 2)

    # De-duplicate all categories.
    if not category:
        for d in os.scandir(IMAGE_DIR):
            if d.is_dir():
                dedup_images(d.name, ssim_threshold, x, y, w, h)
        return

    # Get all images in category.
    files = glob(f"{IMAGE_DIR}/{category}/*.png")
    print(f'Found {len(files)} files in "{category}"')

    title = "Are these duplicates? [y/n]"

    pos = [] # ssim_scores for duplicate images
    neg = [] # ssim_scores for different images

    # Iterate each combination of files.
    for i, file in enumerate(files):
        dup, score, nay = find_dup_image(file, files[i+1:], ssim_threshold, x, y, w, h)

        if dup:
            print(f"Removing duplicate: {file}")
            os.remove(file)
            pos.append(score)
        else:
            neg.extend(nay)

    # Sort ssim scores, so highest score for different images and
    # lowest score for duplicate images are easy to spot.
    pos.sort()
    neg.sort()

    print("Positive counts:")
    print(pos)
    print("False positive counts:")
    print(neg)

def reclassify_images(start_index=None, category=None, ssim_threshold=None, x=None, y=None, w=400, h=200):
    """
    Reclassify mis-classified images.

    The "start_index" is used to divide images in half.

    Set to 0 to build a classifier from even-indexed images; that
    classifier will then search odd-indexed images for mis-classified
    images.

    Set to 1 to build a classifier from odd-indexed images; that
    classifier will then search even-indexed images for mis-classified
    images.

    Set to None, the default, to build both even and odd classifiers
    to search over all images.

    The "category" parameter is the category to check.  The default is
    to check all categories.

    The "ssim_threshold" parameter determines if images are the same.
    The default is SSIM_THRESHOLD.

    The "x", "y", "w" and "h" parameters are for image viewer
    placement.
    """

    if ssim_threshold is None:
        ssim_threshold = SSIM_THRESHOLD

    # Center horizontally.
    if x is None:
        screen_w, screen_h = screen_size()
        x = int((screen_w - w) / 2)

    # Center vertically.
    if y is None:
        screen_w, screen_h = screen_size()
        y = int((screen_h - h) / 2)

    # Check all images.
    if start_index is None:
        reclassify_images(0, category, ssim_threshold, x, y, w, h)  # even
        reclassify_images(1, category, ssim_threshold, x, y, w, h)  # odd
        return

    # Save each half-model in its own file.
    model_file = f"half-model-{start_index}.joblib"

    if os.path.exists(model_file):
        print(f"Loading classifier: {model_file}")
        classifier = joblib.load(model_file)
    else:
        # Slice image files in half, either even or odd indexes.
        even_slicer = slice(start_index, None, 2)
        classifier = build_classifier(slicer=even_slicer, model_file=model_file)

    # Slice other half of image files.
    odd_slicer = slice(start_index ^ 1, None, 2)

    # All image files in each category.
    cat_files = category_files()

    # Category names and index.
    categories = sorted(cat_files.keys())
    cat_index = {cat: i for i, cat in enumerate(categories)}

    # Count [y/n] answers.
    pos = 0
    neg = 0

    for cat, files in cat_files.items():
        # Only process specific category.
        if category and cat != category:
            continue

        # Expected category index.
        index = cat_index[cat]

        for file in files[odd_slicer]:
            # Classify image.
            image = cv2.imread(file)
            flat = [image.flatten()]
            pred = classifier.predict(flat)[0]

            # Image correctly classified.
            if pred == index:
                continue

            # Classifier thinks image is in wrong category.
            new_cat = categories[pred]

            # There will be a lot of false positives to check, but not
            # as bad as checking all images.
            print(f'"{cat}" image might be "{new_cat}": {file}')
            title = f'"{cat}" image should be "{new_cat}"? [y/n]'

            # Is image mis-classified?  Press "N" or "n" if
            # classification is correct.
            key = show_images(image, x=x, y=y, w=w, h=h, title=title)
            if key in 'Nn':
                neg += 1
                continue

            # Image was mis-classified.
            pos += 1

            # Check if duplicate image already exists in new
            # category/directory.
            if is_dup_image(file, cat_files[new_cat], ssim_threshold, x, y, w, h):
                # Duplicate image, so just remove it from its current
                # directory.
                print(f"Removing mis-classified image: {file}")
                os.remove(file)
            else:
                # No duplicate images in new directory, so ok to move.
                new_file = os.path.basename(file)
                new_file = f'{IMAGE_DIR}/{new_cat}/{new_file}'

                print(f"Moving mis-classified image: {file} -> {new_file}")
                os.rename(file, new_file)

    print(f"Positive counts: {pos}")
    print(f"False positive counts: {neg}")

def show_image_count():
    """ Show image count by category. """

    print("Image count:")
    total = 0

    for cat, files in category_files().items():
        count = len(files)
        total += count
        print(f"  {count:4d} {cat}")

    print(f"  {total:4d} TOTAL")

#### Other Functions ####

def add_buttons():
    """ Add "Save Images", "Saving" and "Quit" buttons to captcha. """

    # The "Confirm" and "Loading" buttons:
    #
    #   <div style="display: flex; flex-direction: row; justify-content: center;">
    #   <button type="submit" id="amzn-btn-verify-internal" class="btn btn-primary">Confirm</button>
    #   <button type="button" class="btn btn-primary" style="display: none;">
    #   <img src="data:image/gif;base64..." alt="" role="presentation"> Loading</button>
    #   </div>

    driver.execute_script(
        """
        var confirm_id = arguments[0];

        // Get "Confirm" and "Loading" buttons.
        var confirm = document.getElementById(confirm_id);
        var loading = confirm.nextElementSibling;

        // Create "Save Images" button.
        var save_html = '<button id="zzz-save" type="button" class="btn btn-primary" style="display: none;">Save Images</button>'

        // Create "Saving" button by tweaking copy of "Loading" button
        // to get its loading gif.
        save_html += loading.outerHTML.replace('class=', 'id="zzz-saving" class=').replace('Loading', 'Saving');

        // Create "Quit" button in its own row.
        var quit_html = `
            <div style="display: flex; flex-direction: row; justify-content: center; margin-top: 20px;">
            <button id="zzz-quit" type="button" class="btn btn-primary" style="display: flex; flex-grow: 1; justify-content: center;">Quit</button>
            </div>
        `

        // Add "Save Images" and "Saving" buttons over the "Confirm" button.
        confirm.insertAdjacentHTML('beforebegin', save_html);

        // Add "Quit" button at the end of the form.
        confirm.closest("form").insertAdjacentHTML('beforeend', quit_html);

        // Get button elements.
        var save   = document.getElementById('zzz-save');
        var saving = document.getElementById('zzz-saving');
        var quit   = document.getElementById('zzz-quit');

        // Clicking the "Save Images" button will replace it with
        // "Saving" button.
        save.addEventListener('click', function(e) {
            this.style.display = "none";
            saving.style.display = "inline";
        });

        // Clicking the "Quit" button will nullify any canvas data.
        // It will also hide the "Save Images" button which the web
        // driver is monitoring.
        quit.addEventListener('click', function(e) {
            var canvas = document.getElementsByTagName('canvas')[0];
            canvas.dataset.mimikyu = "QUIT";
            save.style.display = "none";
        });
        """, confirm_button_id()
    )

def begin_page():
    """ Navigate captcha intro. """

    # The captcha starts with a brief intro.  Click the "Begin >"
    # button to continue to puzzle:
    #
    #   <button class="amzn-captcha-verify-button btn btn-primary" id="amzn-captcha-verify-button" type="button" style="display: flex; padding: 5px 30px;">
    #   Begin<img src="data:image/svg+xml;base64,..." alt="begin" style="margin-left: 5px;">
    #   </button>

    begin = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.XPATH, "//button[@id='amzn-captcha-verify-button']"))
    )

    # Pause a bit.
    pause(0.5, 1.5)

    begin.click()

    wait_for_puzzle()

def canvas_area():
    """ Get canvas location (relative to whole screen) and size. """

    # Height of title, toolbar, etc on top of browser, assuming no
    # status bar at bottom.
    top_height = driver.execute_script("return window.outerHeight - window.innerHeight")

    canvas = canvas_element()

    # Canvas location relative to HTML page.
    loc = canvas.location

    # Browser location relative to whole screen.
    pos = driver.get_window_position()

    # Canvas location relative to whole screen.
    x = pos.get('x') + loc['x']
    y = pos.get('y') + loc['y'] + top_height

    # Canvas size.
    size = canvas.size
    w = size['width']
    h = size['height']

    return x, y, w, h

def canvas_element():
    """ Get canvas element. """

    return driver.find_element(By.XPATH, "//canvas")

def category_files():
    """ Return all image files by category. """

    cat_files = defaultdict(list)

    for file in glob(f"{IMAGE_DIR}/**/*.png"):
        dir = os.path.dirname(file)
        cat = os.path.basename(dir)
        cat_files[cat].append(file)

    return cat_files

def click_tile(ij):
    """ Click an image tile in puzzle. """

    # Coordinates as integers.
    i, j = (int(x) for x in ij)

    # Canvas location relative to HTML page.
    canvas = canvas_element()
    loc = canvas.location
    x = loc['x']
    y = loc['y']

    # Tile location relative to canvas.
    l, t, r, b = IMAGE_CROP[ij]

    # Tile location relative to HTML page.
    x += l
    y += t

    # Center of tile.
    half = IMAGE_SIZE / 2
    x += half
    y += half

    # Remove 5px margin to avoid clicking edge of image.
    half -= 5

    # Random distance from center of tile.
    x += int(half * randomish())
    y += int(half * randomish())

    # Pause a bit.
    pause(0.5, 1.5)

    # Click tile.
    action = ActionBuilder(driver)
    action.pointer_action.move_to_location(x, y)
    action.pointer_action.click()
    action.perform()

def confirm_button_id():
    """ Get "Confirm" button ID. """

    # The "Confirm" button:
    #
    #   <button type="submit" id="amzn-btn-verify-internal" class="btn btn-primary" style="display: block;">Confirm</button>

    return 'amzn-btn-verify-internal'

def find_dup_image(dup, files, ssim_threshold, x, y, w, h):
    """
    Find first file in "files" that is a duplicate image of "dup".

    Returns a tuple of found file, its ssim_score, and a list of false
    positive scores.

    The "ssim_threshold" parameter determines if images are the same.

    The "x", "y", "w" and "h" parameters are for image viewer
    placement.
    """

    title = "Are these duplicates? [y/n]"

    # ssim_scores for false positives.
    nay = []

    for file in files:
        score = ssim_score(dup, file)

        # Different images.
        if score < ssim_threshold:
            continue

        # Duplicate images.
        score = "%.02f" % score

        print(f"Found duplicate images with score={score}:")
        if type(dup)  is str: print("    " + dup)
        if type(file) is str: print("    " + file)

        # A pixel-perfect duplicate.  No need to ask user.
        if score == '1.00':
            return file, score, nay

        # Show duplicate images.  Press "N" or "n" if images are
        # different.
        key = show_images(dup, file, x=x, y=y, w=w, h=h, title=title)

        if key in 'Nn':
            nay.append(score)
        else:
            return file, score, nay

    return None, None, nay

# Store classifier once built or loaded.
CLASSIFIER = None 

def get_classifier():
    """
    Get classifier.  Load from file or build from images if
    necessary.
    """

    global CLASSIFIER

    # Return in-memory classifier.
    if CLASSIFIER:
        return CLASSIFIER

    # Load classifier from file.
    CLASSIFIER = load_classifier()
    if CLASSIFIER:
        return CLASSIFIER

    # Build classifier from images.
    CLASSIFIER = build_classifier()
    return CLASSIFIER

def get_new_puzzle():
    """ Get a new puzzle. """

    # Hide whichever button is in the "Confirm" button spot, so there
    # is room when the new puzzle makes the "Confirm" button visible
    # again.
    driver.execute_script(
        """
        for (var id of arguments) {
            var elem = document.getElementById(id);
            if (elem) {
                elem.style.display = "none";
            }
        }
        """, confirm_button_id(), "zzz-save", "zzz-saving"
    )

    # Click "Get a new puzzle" button:
    #
    #   <button type="button" id="amzn-btn-refresh-internal" class="btn-icon" disabled="">
    #   <img src="data:image/svg+xml;base64..." alt="Get a new puzzle">
    #   </button>

    driver.find_element(By.XPATH, "//button[@id='amzn-btn-refresh-internal']").click()

    wait_for_puzzle()

def grid_search_best_params(param_grid=None, max_images=None):
    """
    Run GridSearchCV() to get best parameters.

    The "param_grid" parameter is a grid of hyper-paramters used by
    GridSearchCV() to find the best paramters.

    The "max_images" parameter is used to limit the number of images
    per category.  Too few images will generate warnings, while too
    many images will require more time:

      max_images  elapsed_time
               5  0:01:23.612069
              10  0:04:59.285925
              15  0:10:26.207704
              20  0:18:10.407012
    """

    # Import modules here since they take a while to load.
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV

    # Get slice() for max_images.
    slicer = slice(0, max_images) if max_images else None

    # Slice of image files by category.
    cat_files = slice_category_files(slicer)

    # Load images into numerical arrays.
    X, y = load_image_data(cat_files)

    # Grid search for best parameters.
    svc = svm.SVC(probability=True)
    if not param_grid:
        param_grid = {
            'C':      [0.1, 1, 10, 100],
            'gamma':  [0.0001, 0.001, 0.1, 1],
            'kernel': ['rbf', 'poly'],
        }
    model = GridSearchCV(svc, param_grid)

    # Fit data to model.
    start_dt = datetime.now()
    print("Start grid search:", start_dt)
    model.fit(X, y)
    print("Finish grid search, elapsed time:", datetime.now() - start_dt)

    # Save best_params_ to file.
    best = model.best_params_
    print("best_params_ =", best)
    joblib.dump(best, BEST_PARAMS_FILE)

    return best

def initiate_puzzle():
    """ Initiate new puzzle. """

    # Each new puzzle makes the "Confirm" button visible.
    #
    # Each new puzzle creates a brand new canvas, so canvas dataset
    # and listener are not persistent.

    driver.execute_script(
        """
        var confirm_id = arguments[0];
        var CLICK_SIZE = arguments[1];

        // Replace "Confirm" button with "Save Images" button.
        document.getElementById(confirm_id).style.display = "none";
        document.getElementById('zzz-save').style.display = "inline";

        // Add listener to copy user clicks on canvas.
        var canvas = document.getElementsByTagName('canvas')[0];
        canvas.dataset.mimikyu = "";
        canvas.addEventListener('click', function(e) {
            var x = Math.floor(e.offsetX / CLICK_SIZE);
            var y = Math.floor(e.offsetY / CLICK_SIZE);
            this.dataset.mimikyu += `${x}${y} `;
        });
        """, confirm_button_id(), CLICK_SIZE
    )

def instruction_cat():
    """ Get category from instructions. """

    # Instructions on which images to pick:
    #
    #   <div style="margin-bottom: 0.5em;">
    #   Choose all <em style="font-weight: bold; text-decoration: underline; font-style: normal;">the clocks</em>
    #   </div>

    em = driver.find_element(By.XPATH, "//div[contains(text(), 'Choose all')]/em")
    return em.text.split()[-1]

def is_dup_image(dup, files, ssim_threshold, x, y, w, h):
    """ Returns whether there is a duplicate image of "dup" in "files" """

    return find_dup_image(dup, files, ssim_threshold, x, y, w, h)[0]

def load_classifier():
    """ Load classifier from file. """

    global CLASSIFIER

    # Return in-memory classifier.
    if CLASSIFIER:
        return CLASSIFIER

    # Load from file.
    if os.path.exists(MODEL_FILE):
        print(f"Loading classifier: {MODEL_FILE}")
        CLASSIFIER = joblib.load(MODEL_FILE)
        return CLASSIFIER

    return None

def load_image_data(cat_files):
    """ Load images into numerical arrays. """

    # Category index, since model operates on numerical arrays.
    categories = sorted(cat_files.keys())
    cat_index = {cat: i for i, cat in enumerate(categories)}

    # Feature matrix of flattened images.
    feature = []

    # Target vector of category indexes.
    target = []

    # Load images.
    for cat, files in cat_files.items():
        index = cat_index[cat]
        count = len(files)

        print(f"Loading category: {count:3d} {cat}")

        for file in files:
            image = cv2.imread(file)
            feature.append(image.flatten())
            target.append(index)

    # Numpy arrays.
    X = np.array(feature)
    y = np.array(target)

    return X, y

def pause(min=1, max=None):
    """ Pause between min and max seconds. """

    seconds = min

    if max and max > min:
        # Half the distance between min and max.
        half = (max - min ) / 2

        # Average of min and max.
        seconds += half

        # Random number from bell curve between min and max.
        seconds += half * randomish()

    time.sleep(seconds)

def randomish():
    """
    Get random number from a normal distribution (bell curve) but
    limited to values between -1 and 1.
    """

    while True:
        # Random number from a bell curve centered at 0 with standard
        # deviation of 0.5.
        r = np.random.normal(0, 0.5)

        # Limiting values to +/- 1.  This is the 2nd standard deviation
        # or 95% of all values.
        if -1 <= r <= 1:
            return r

def reload_captcha():
    """ Reload captcha. """

    print("Reloading captcha.")

    # Browser reload goes back to the captcha intro.
    driver.refresh()

    # Navigate captcha intro back to captcha puzzle.
    begin_page()    

# Store screen width and height so only calculated once.
SCREEN_WIDTH  = None
SCREEN_HEIGHT = None

def screen_size():
    """ Return screen width and height. """

    global SCREEN_WIDTH, SCREEN_HEIGHT

    if SCREEN_WIDTH:
        return SCREEN_WIDTH, SCREEN_HEIGHT

    root = tkinter.Tk()
    SCREEN_WIDTH  = root.winfo_screenwidth()
    SCREEN_HEIGHT = root.winfo_screenheight()
    root.destroy()

    return SCREEN_WIDTH, SCREEN_HEIGHT

def screenshot_canvas():
    """ Get screenshot of canvas element. """

    file = IMAGE_DIR + "/canvas.png"

    # Screenshot to file.
    canvas_element().screenshot(file)

    # Read file.  This converts image to OpenCV BGR format used
    # everywhere else.
    return cv2.imread(file)

def show_images(*images, **kw):
    """
    Show images horizontally.  Returns user key press. 

    The "images" parameter is a list of images or image files.

    The "kw" parameter is for the image viewer:  title, x, y, w, h
    """

    # Concatenate all images horizontally.
    for i, img in enumerate(images):
        if type(img) is str:
            img = cv2.imread(img)
        image = img if i == 0 else cv2.hconcat([image, img])

    # Image viewer title.
    title = kw.get('title', f"{len(images)} images")

    # Image viewer location.
    x = kw.get('x', 0)
    y = kw.get('y', 0)

    # Image viewer size.
    h, w, _ = image.shape
    h = max(h, kw.get('h', 0))
    w = max(w, kw.get('w', 0))

    # Show image.
    cv2.imshow(title, image)
    cv2.moveWindow(title, x, y)
    cv2.resizeWindow(title, w, h)

    # If there's a web driver, minimize the browser window to
    # relinquish focus, so it goes to image viewer.
    if driver:
        driver.minimize_window()

    # Wait for user key press before closing window.
    key = chr(cv2.waitKey(0))
    cv2.destroyWindow(title)
    return key

def show_puzzle_solution(cat, data):
    """ Show puzzle solution. """

    w       = 14                             # Column width.
    cat_td  = "{:s}{:^%ds}|" % (w - 1)       # Format for match/category cell.
    prob_td = " {:^%d.1%%}|" % (w - 1)       # Format for probability cell.
    cat_tr  = "    |" + cat_td * 3           # Format for match/category row.
    prob_tr = "    |" + prob_td * 3          # Format for probability row.
    hr      = "    +" + ("-" * w + "+") * 3  # Horizontal border.

    print(f'Image classifications and "{cat}" probabilities:')

    # Iterate one row (3 keys) at a time.
    it = iter(IMAGE_CROP.keys())
    for row in zip(it, it, it):
        cats  = []  # Match and category.
        probs = []  # Probabilities.

        # Collect data in each row.
        for ij in row:
            td = data[ij]
            cats.append("*" if td["match"] else " ")
            cats.append(td["cat"])
            probs.append(td["prob"])

        print(hr)
        print(cat_tr.format(*cats))
        print(prob_tr.format(*probs))

    print(hr)

def slice_category_files(slicer, cat_files=None):
    """ Return a slice() of the category_files() """

    if not cat_files:
        cat_files = category_files()
    if not slicer:
        return cat_files

    return {cat: files[slicer] for cat, files in cat_files.items()}

def ssim_score(image1, image2):
    """
    Return structural similarity score.  Similar images will score
    closer to 1.
    """

    # Load images if file names.
    if type(image1) is str: image1 = cv2.imread(image1)
    if type(image2) is str: image2 = cv2.imread(image2)

    # Compare images by pixel.
    if np.all(image1 == image2):
        return 1

    # Convert images to grayscale.
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    return ssim(image1, image2)

# Store titlebar_height() to avoid calculating again.
TITLEBAR_HEIGHT = None

def titlebar_height(x=1, y=1, w=1, h=1):
    """
    Get title bar height of a Tk widget.

    Even though cv2 image viewer doesn't use Tk, it should have
    the same or similar height.
    """

    global TITLEBAR_HEIGHT;
    if TITLEBAR_HEIGHT is not None:
        return TITLEBAR_HEIGHT

    root = tkinter.Tk()
    root.geometry(f"{w}x{h}")   # Size
    root.geometry(f"+{x}+{y}")  # Location
    root.update()
    top_of_browser  = root.winfo_rooty()
    top_of_screen   = root.winfo_y()
    TITLEBAR_HEIGHT = top_of_browser - top_of_screen
    root.destroy()

    return TITLEBAR_HEIGHT

def wait_for_puzzle():
    """ Wait for puzzle to load. """

    # Not just for the first puzzle, but whenever a new puzzle is
    # loaded, the "Confirm" button is made visible.

    id = confirm_button_id()
    WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.XPATH, f"//button[@id='{id}']"))
    )

# Global web driver is used everywhere.
driver = None

def web_driver(set_driver):
    """
    Get a web driver.  Defaults to Chrome. 

    The "set_driver" parameter will set the driver.
    """

    global driver

    if set_driver:
        driver = set_driver
        return driver

    # Create driver with Chrome.
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(options=chrome_options)
    return driver




if __name__ == "__main__":

    # Parse commandline arguments.
    parser = argparse.ArgumentParser(
        prog='captcha',
        description='Pokemon Cafe captcha')

    parser.add_argument(
        '-c', '--collect-images',
        action='store_true',
        help='Collect captcha images.',
    )
    parser.add_argument(
        '-b', '--build-classifier',
        action='store_true',
        help='Build classifier from captcha images.',
    )
    parser.add_argument(
        '-s', '--solve-captcha',
        action='store_true',
        help='Solve captcha.',
    )
    parser.add_argument(
        '-d', '--dedup-images',
        action='store_true',
        help='De-duplicate identical images.',
    )
    parser.add_argument(
        '-r', '--reclassify-images',
        action='store_true',
        help='Reclassify mis-classified images.',
    )
    parser.add_argument(
        '-i', '--image-count',
        action='store_true',
        help='Count collected images.',
    )
    parser.add_argument(
        '-u', '--url',
        default='https://reserve.pokemon-cafe.jp/reserve/step1',
        help='Captcha URL. Defaults to "%(default)s"',
    )
    parser.add_argument(
        '-C', '--category',
        help='Image category.',
    )
    parser.add_argument(
        '-S', '--ssim-threshold',
        type=float,
        help='Threshold for duplicate images.',
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Testing.',
    )

    args = parser.parse_args()

    if args.collect_images:
        print("Collecting images...")
        collect_images(url=args.url, ssim_threshold=args.ssim_threshold, test=args.test)
    elif args.build_classifier:
        print("Building classifier...")
        build_classifier(test=args.test)
    elif args.solve_captcha:
        print("Solving captcha...")
        solve_captcha(url=args.url, test=args.test)
    elif args.dedup_images:
        print("Deduping images...")
        dedup_images(category=args.category, ssim_threshold=args.ssim_threshold)
    elif args.reclassify_images:
        print("Reclassifying images...")
        reclassify_images(category=args.category, ssim_threshold=args.ssim_threshold)
    elif args.image_count:
        show_image_count()
    else:
        parser.print_help()



# Captcha HTML form:
#
#   <form>
#   <div class="amzn-captcha-modal-title" style="font-size: 1.5em; margin-bottom: 20px;">Let's confirm you are human</div>
#   <div>
#   </div>
#   <div style="margin-bottom: 10px;">
#   <div tabindex="-1" style="outline-color: rgba(0, 0, 0, 0);">
#   <div style="margin-bottom: 0.5em;">Choose all <em style="font-weight: bold; text-decoration: underline; font-style: normal;">the buckets</em>
#   </div>
#   <div style="position: relative;">
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
#   <div class="amzn-grid-focus" style="width: 104px; height: 104px; display: none;">
#   </div>
#   </div>
#   </div>
#   </div>
#   <div style="display: none;">Solved: <span>0</span> Required: <span>1</span>
#   </div>
#   <div role="alert" tabindex="-1" style="margin-bottom: 20px; display: none; background-color: rgb(255, 233, 228); padding: 5px 20px; font-size: 0.875em; outline: red solid 2px;">
#   <img src="data:image/svg+xml;base64,...+IA==" role="presentation" alt="">
#   <p style="margin-left: 10px;">Time limit exceeded. Please refresh the page.</p>
#   </div>
#   <div id="amzn-help-container" aria-expanded="false" tabindex="-1" style="overflow: hidden; background-color: rgb(242, 243, 243); transition: max-height 500ms ease-in; max-height: 0px; font-size: 0.875em; display: none;">
#   <div style="flex: 1 1 0%; padding: 0px 20px;">
#   <div style="display: flex;">
#   <p style="padding-right: 1em;">Choose only the images that contain the underlined object in the instructions. You can choose the image(s) by tapping them and then Confirm to submit your answer.</p>
#   </div>
#   </div>
#   <div style="width: 24px;">
#   <button aria-controls="amzn-help-container" type="button" style="border: 0px; padding: 0px; cursor: pointer;">
#   <img src="data:image/svg+xml;base64,...+IDwvc3ZnPiA=" alt="Close help">
#   </button>
#   </div>
#   </div>
#   <div style="display: flex; flex-direction: row; justify-content: space-between; margin-top: 20px;">
#   <div style="display: flex; flex-direction: row; justify-content: center;">
#   <button type="button" id="amzn-btn-refresh-internal" class="btn-icon">
#   <img src="data:image/svg+xml;base64,...+IA==" alt="Get a new puzzle">
#   </button>
#   <button type="button" id="amzn-btn-info-internal" aria-controls="amzn-help-container" class="btn-icon">
#   <img src="data:image/svg+xml;base64,..." alt="Instructions">
#   </button>
#   <button type="button" id="amzn-btn-audio-internal" class="btn-icon">
#   <img src="data:image/svg+xml;base64,...+IDwvc3ZnPiA=" alt="Get an audio puzzle">
#   </button>
#   <div style="position: fixed; z-index: 20; top: 458px; display: none; left: 387px;">
#   <div style="background-color: rgb(22, 25, 31); color: rgb(250, 250, 250); padding: 9px 14px;">Get an audio puzzle</div>
#   <div style="width: 0px; height: 0.875em; border-top: 0.875em solid rgb(22, 25, 31); border-right: 0.875em solid transparent;">
#   </div>
#   </div>
#   </div>
#   <div style="display: flex; flex-direction: row; justify-content: center;">
#   <button type="submit" id="amzn-btn-verify-internal" class="btn btn-primary" style="display: block;">Confirm</button>
#   <button type="button" class="btn btn-primary" style="display: none;">
#   <img src="data:image/gif;base64,..." alt="" role="presentation"> Loading</button>
#   </div>
#   </div>
#   <div style="display: flex; flex-direction: row; justify-content: space-between; margin-top: 5px; min-height: 24px;">
#   <select class="amzn-captcha-lang-selector" aria-label="Select language">
#   <option value="ar" aria-labelledby="ar">العربية</option>
#   <option value="cs" aria-labelledby="cs">Čeština</option>
#   <option value="da" aria-labelledby="da">Dansk</option>
#   <option value="de" aria-labelledby="de">Deutsch</option>
#   <option value="en" aria-labelledby="en" selected="">English</option>
#   <option value="es" aria-labelledby="es">Español</option>
#   <option value="fr" aria-labelledby="fr">Français</option>
#   <option value="it" aria-labelledby="it">Italiano</option>
#   <option value="ja" aria-labelledby="ja">日本語</option>
#   <option value="nl" aria-labelledby="nl">Nederlands</option>
#   <option value="pl" aria-labelledby="pl">Polski</option>
#   <option value="pt" aria-labelledby="pt">Português</option>
#   <option value="sv" aria-labelledby="sv">Svenska</option>
#   <option value="tr" aria-labelledby="tr">Türkçe</option>
#   <option value="zh" aria-labelledby="zh">中文</option>
#   </select>
#   </div>
#   </form>


