import re
import cv2
import difflib
import itertools
import numpy as np
import logging
import sys


def enhance_image(crop_image, alpha=1.5, beta=0):
    gray_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_image, (5, 5), 0)
    adpt_img = cv2.adaptiveThreshold(blur_img, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
    return adpt_img


def get_ConfigLogger(debug=False):
    logFormatter = logging.Formatter("%(asctime)-15s [%(levelname)-5.5s]%(message)s", "%Y-%m-%d %H:%M:%S")
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    rootLogger = logging.getLogger()

    # define Console Handler
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    return 0


def emirate_check(text_en):
    city = None
    # find closest match to ABU DHABI

    if difflib.get_close_matches('ABUDHABI', text_en) or 'AD' in text_en or difflib.get_close_matches('UAEAD', text_en):
        city = "AD"
    # if closest match to SHarjah
    elif difflib.get_close_matches('SHARJH', text_en):
        city = "SHARJAH"
    elif difflib.get_close_matches('AJMAN', text_en) or difflib.get_close_matches('UAEAJMAN', text_en):
        city = "AJMAN"
    elif 'RAK' in text_en:
        city = "RAS AL KHAIMAH"
    elif 'UAQ' in text_en:
        city = "UAQ"
    logging.info("city {}".format(city))
    return city


def filter_eng_data(text_data):
    text_data = [re.findall(r'[A-Z]+', text) for text in text_data]
    text_data = list(itertools.chain(*text_data))
    filter_text = list(filter(str.strip, text_data))

    digit_text = [re.findall(r'[0-9]+', text) for text in text_data]
    digit_text = list(itertools.chain(*digit_text))
    digit_text = list(filter(str.strip, digit_text))

    return filter_text, digit_text


def filter_arabic_data(arabic_text):
    # filter Arabic text
    text_ar = [r[1].translate({ord(i): None for i in "':!?+|\/}{*%&#()$-_=[]^., "}) for r in arabic_text if
               not any(c.isdigit() for c in r[1])]

    digit_ar = [r[1].translate({ord(i): None for i in "':!?+|\/}{*%&#()$-_=[]^., "}) for r in arabic_text if
                any(c.isdigit() for c in r[1])]
    digit_ar = [re.findall(r'[0-9]+', text) for text in digit_ar]
    digit_ar = list(itertools.chain(*digit_ar))
    digit_ar = list(filter(str.strip, digit_ar))
    return text_ar, digit_ar


def evaluate_data(results_en, results_ar):
    plate_state = list()
    num_string = list()
    unfilterted_data = list()

    # filter Text data
    en_results = [r[1] for r in results_en]
    text_en, digit_en = filter_eng_data(en_results)
    logging.info("Text en:{}".format(text_en))
    logging.info("Digit en {}".format(digit_en))
    # add unfiltered data
    unfilterted_data.append(text_en)
    unfilterted_data.append(digit_en)

    # filter Arabic text
    text_ar, digit_ar = filter_arabic_data(results_ar)
    logging.info("Text ar :{}".format(text_ar))
    logging.info("Digit ar:{}".format(digit_ar))
    # add unfiltered data
    unfilterted_data.append(text_ar)
    unfilterted_data.append(digit_ar)
    # check for emirate in textual data of English
    # find closest match to DUBAI
    emirate = None
    if 'DUBAI' in text_en:
        emirate = 'DUBAI'
        plate_state.append('DUBAI')
    else:
        emirate = emirate_check(text_en)
        if emirate is not None:
            plate_state.append(emirate)

    if emirate == 'AD' or emirate == 'SHARJAH':
        ar_num = [num for num in digit_ar if len(num) in range(1, 6)]
        en_num = [num for num in digit_en if len(num) in range(1, 6)]
        num_string = ar_num + list(set(en_num) - set(ar_num))

    elif emirate == 'DUBAI' or emirate == 'SHARJAH' or emirate == 'AJMAN' or emirate == 'RAK' or emirate == 'UAQ':
        ar_num = [num for num in digit_ar if len(num) in range(1, 6)]
        en_num = [num for num in digit_en if len(num) in range(1, 6)]
        cha_string = [ch for ch in text_en if len(ch) == 1]
        num_string = cha_string + ar_num + list(set(en_num) - set(ar_num))
    else:
        cha_string = [ch for ch in text_en if len(ch) in range(1, 2)]
        num_char = digit_ar + list(set(digit_en) - set(digit_ar))
        num_string = num_char + cha_string
        # num_string = list(filter(str.strip,num_string))
        # if emirate is None and len(en_num) == 0:
        #     diff = list(set(text_en).symmetric_difference(num_string))
        #     num_string = text_en + diff

    # find numeric data from english text
    ocr_data = num_string + plate_state
    ocr_data = list(ocr_data + list(set(text_ar) - set(ocr_data)))
    unfilterted_data = list(np.concatenate(unfilterted_data))
    return ocr_data, unfilterted_data
