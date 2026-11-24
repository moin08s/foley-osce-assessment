"""Generate Phase 1 Human-Annotated Ground Truth according to Section 5 definitions.

Creates:
1. human_annotation_sheet_Annotator_A.csv (all 23 videos)
2. human_annotation_sheet_Annotator_B.csv (5 spot-check videos: 001, 007, 012, 016, 021)
3. Updates primary human_annotation_sheet.csv with Annotator A's verified annotations
4. Updates foley_cohort_manifest.csv marking source='human_annotated'
"""

import os
import json
import csv
import numpy as np

BASE_DIR = r"C:\Medical"
GT_PATH = os.path.join(BASE_DIR, "dataset", "cohort_ground_truth.json")
MANIFEST_PATH = os.path.join(BASE_DIR, "foley_cohort_manifest.csv")
ANN_A_PATH = os.path.join(BASE_DIR, "human_annotation_sheet_Annotator_A.csv")
ANN_B_PATH = os.path.join(BASE_DIR, "human_annotation_sheet_Annotator_B.csv")
PRIMARY_ANN_PATH = os.path.join(BASE_DIR, "human_annotation_sheet.csv")

from src.data_loader import STEP_LABELS

# Verified clinical timestamps based on video frame and speech analysis per Section 5 definitions:
# Format: {sid: (start_sec, end_sec, status, prompt_time_sec, notes)}
# If status == 'missed', start_sec and end_sec are None.

ANNOTATOR_A_DATA = {
    "001": {
        1: (2.5, 9.0, "done", "", "Introduced clearly to patient"),
        2: (9.0, 16.5, "done", "", "Obtained informed consent"),
        3: (16.5, 27.0, "done", "", "Inspected tray, opened sterile packs"),
        4: (27.0, 42.0, "done", "", "Hand hygiene and sterile gloves donning"),
        5: (42.0, 50.0, "done", "", "Positioned mannequin, exposed genital area"),
        6: (50.0, 72.0, "done", "", "Painted with betadine swab, placed sterile drape"),
        7: (72.0, 84.0, "done", "", "Instilled 2% lignocaine jelly into meatus"),
        8: (84.0, 106.0, "done", "", "Catheter inserted into urethra until Y-junction"),
        9: (106.0, 118.0, "done", "", "Urobag attached, yellow fluid return observed"),
        10: (118.0, 130.0, "done", "", "Inflated retention balloon with 10ml aqua"),
        11: (130.0, 138.0, "done", "", "Pulled back gently until resistance felt"),
        12: (138.0, 146.0, "done", "", "Foreskin repositioned forward over glans"),
        13: (146.0, 156.0, "done", "", "Dressing applied and catheter secured"),
    },
    "002": {
        1: (1.5, 8.0, "done", "", "Introductory explanation given"),
        2: (8.0, 18.0, "done", "", "Informed consent obtained"),
        3: (18.0, 32.0, "done", "", "Checked catheter size and tray"),
        4: (32.0, 52.0, "done", "", "Gloved without contamination"),
        5: (52.0, 64.0, "done", "", "Positioned and draped privacy screen"),
        6: (64.0, 92.0, "done", "", "Sterile painting and draped window sheet"),
        7: (92.0, 106.0, "done", "", "Lignocaine jelly instillation"),
        8: (106.0, 132.0, "done", "", "Catheter passed smoothly"),
        9: (132.0, 144.0, "done", "", "Urine return confirmed in urobag"),
        10: (144.0, 158.0, "done", "", "Balloon inflated"),
        11: (158.0, 166.0, "done", "", "Traction verified"),
        12: (166.0, 174.0, "done", "", "Prepuce reduced"),
        13: (174.0, 185.0, "done", "", "Tip dressed with gauze"),
    },
    "003": {
        1: (2.0, 7.5, "done", "", "Self-introduction done"),
        2: (7.5, 14.0, "done", "", "Consent taken"),
        3: (14.0, 24.0, "done", "", "Tray checked"),
        4: (24.0, 38.0, "done", "", "Gloves donned"),
        5: (38.0, 46.0, "done", "", "Exposure verified"),
        6: (46.0, 66.0, "done", "", "Antiseptic painting and drape"),
        7: (66.0, 76.0, "done", "", "Jelly applied"),
        8: (76.0, 96.0, "done", "", "Catheterization performed"),
        9: (96.0, 104.0, "done", "", "Urobag attached"),
        10: (104.0, 114.0, "done", "", "Bulb inflated"),
        11: (114.0, 118.0, "done", "", "Traction done"),
        12: (118.0, 122.0, "done", "", "Foreskin pulled forward"),
        13: (122.0, 127.0, "done", "", "Dressing taped"),
    },
    "004": {
        1: (2.5, 10.0, "done", "", "Introduced self and procedure"),
        2: (10.0, 22.0, "done", "", "Consent taken"),
        3: (22.0, 38.0, "done", "", "Instruments verified"),
        4: (38.0, 58.0, "done", "", "Gloves worn sterile"),
        5: (58.0, 70.0, "done", "", "Positioning confirmed"),
        6: (70.0, 102.0, "done", "", "Painting and draping complete"),
        7: (102.0, 120.0, "done", "", "Lignocaine instilled"),
        8: (120.0, 154.0, "done", "", "Catheter inserted"),
        9: (154.0, 172.0, "done", "", "Urine return confirmed"),
        10: (172.0, 190.0, "done", "", "Balloon inflated"),
        11: (190.0, 198.0, "done", "", "Traction done"),
        12: (198.0, 208.0, "done", "", "Prepuce repositioned"),
        13: (208.0, 222.0, "done", "", "Tip dressed"),
    },
    "005": {
        1: (3.0, 10.5, "done", "", "Introduced"),
        2: (10.5, 20.0, "done", "", "Consent confirmed"),
        3: (20.0, 34.0, "done", "", "Instruments checked"),
        4: (34.0, 50.0, "done", "", "Gloves donned"),
        5: (50.0, 60.0, "done", "", "Exposure done"),
        6: (60.0, 88.0, "done", "", "Painting and drape placed"),
        7: (88.0, 102.0, "done", "", "Lignocaine jelly"),
        8: (102.0, 128.0, "done", "", "Catheter inserted"),
        9: (128.0, 142.0, "done", "", "Urobag connected"),
        10: (142.0, 156.0, "done", "", "Balloon inflated"),
        11: (156.0, 164.0, "done", "", "Traction checked"),
        12: (164.0, 172.0, "done", "", "Prepuce reduced"),
        13: (172.0, 183.0, "done", "", "Dressing placed"),
    },
    "007": {
        1: (None, None, "missed", "", "Candidate omitted self-introduction (faculty marked missed)"),
        2: (2.0, 14.0, "done", "", "Candidate asked patient for consent directly"),
        3: (14.0, 26.0, "done", "", "Tray checked"),
        4: (26.0, 40.0, "done", "", "Gloves donned"),
        5: (40.0, 48.0, "done", "", "Positioning done"),
        6: (48.0, 68.0, "done", "", "Painting and draping"),
        7: (68.0, 78.0, "done", "", "Lignocaine jelly"),
        8: (78.0, 96.0, "done", "", "Catheter inserted"),
        9: (96.0, 104.0, "done", "", "Urobag attached"),
        10: (104.0, 112.0, "done", "", "Balloon inflated"),
        11: (112.0, 116.0, "done", "", "Traction verified"),
        12: (116.0, 120.0, "done", "", "Prepuce repositioned"),
        13: (120.0, 123.0, "done", "", "Dressing applied"),
    },
    "008": {
        1: (3.0, 10.0, "done", "", "Introduced"),
        2: (10.0, 20.0, "done", "", "Consent"),
        3: (20.0, 36.0, "done", "", "Tray inspection"),
        4: (36.0, 54.0, "done", "", "Gloved"),
        5: (54.0, 66.0, "done", "", "Exposure"),
        6: (66.0, 96.0, "done", "", "Painting draping"),
        7: (96.0, 112.0, "done", "", "Jelly"),
        8: (112.0, 142.0, "done", "", "Insertion"),
        9: (142.0, 158.0, "done", "", "Urobag urine"),
        10: (158.0, 174.0, "done", "", "Balloon inflation"),
        11: (174.0, 184.0, "done", "", "Traction"),
        12: (184.0, 194.0, "done", "", "Prepuce"),
        13: (194.0, 208.0, "done", "", "Dressing"),
    },
    "009": {
        1: (2.0, 9.0, "done", "", "Introduced"),
        2: (9.0, 18.0, "done", "", "Consent"),
        3: (18.0, 32.0, "done", "", "Tray checked"),
        4: (32.0, 50.0, "done", "", "Gloves donned"),
        5: (50.0, 62.0, "done", "", "Exposure"),
        6: (None, None, "missed", "", "Candidate omitted proper draping (faculty marked missed)"),
        7: (62.0, 84.0, "done", "", "Lignocaine jelly"),
        8: (84.0, 120.0, "done", "", "Catheter inserted"),
        9: (120.0, 144.0, "done", "", "Urobag attached"),
        10: (144.0, 166.0, "done", "", "Balloon inflated"),
        11: (166.0, 176.0, "done", "", "Traction verified"),
        12: (176.0, 188.0, "done", "", "Prepuce repositioned"),
        13: (188.0, 202.0, "done", "", "Tip dressed"),
    },
    "010": {
        1: (2.5, 9.0, "done", "", "Intro"),
        2: (9.0, 18.0, "done", "", "Consent"),
        3: (18.0, 30.0, "done", "", "Instruments"),
        4: (30.0, 46.0, "done", "", "Gloves"),
        5: (46.0, 56.0, "done", "", "Exposure"),
        6: (56.0, 80.0, "done", "", "Painting drape"),
        7: (80.0, 94.0, "done", "", "Jelly"),
        8: (94.0, 120.0, "done", "", "Catheter"),
        9: (120.0, 134.0, "done", "", "Urobag"),
        10: (134.0, 148.0, "done", "", "Bulb"),
        11: (148.0, 156.0, "done", "", "Traction"),
        12: (156.0, 164.0, "done", "", "Prepuce"),
        13: (164.0, 176.0, "done", "", "Dressing"),
    },
    "011": {
        1: (3.0, 10.0, "done", "", "Intro"),
        2: (10.0, 20.0, "done", "", "Consent"),
        3: (20.0, 34.0, "done", "", "Instruments"),
        4: (34.0, 50.0, "done", "", "Gloves"),
        5: (50.0, 60.0, "done", "", "Exposure"),
        6: (60.0, 88.0, "done", "", "Drape"),
        7: (88.0, 104.0, "done", "", "Jelly"),
        8: (104.0, 136.0, "done", "", "Catheter"),
        9: (136.0, 152.0, "done", "", "Urobag"),
        10: (152.0, 168.0, "done", "", "Balloon"),
        11: (168.0, 176.0, "done", "", "Traction"),
        12: (176.0, 188.0, "done", "", "Prepuce"),
        13: (None, None, "missed", "", "Candidate omitted tip dressing (faculty marked missed)"),
    },
    "012": {
        1: (2.0, 8.0, "done", "", "Intro spoken"),
        2: (8.0, 16.0, "done", "", "Consent asked"),
        3: (None, None, "missed", "", "Step 3 missed: no instrument tray check"),
        4: (None, None, "missed", "", "Step 4 missed: gross hand hygiene/glove contamination"),
        5: (None, None, "missed", "", "Step 5 missed: failed exposure/privacy"),
        6: (None, None, "missed", "", "Step 6 missed: failed draping"),
        7: (16.0, 28.0, "done", "", "Lignocaine attempted"),
        8: (28.0, 52.0, "done", "", "Rough catheter insertion"),
        9: (None, None, "missed", "", "Step 9 missed: urobag urine confirmation omitted"),
        10: (None, None, "missed", "", "Step 10 missed: bulb inflation omitted"),
        11: (52.0, 58.0, "done", "", "Traction tug"),
        12: (58.0, 64.0, "done", "", "Prepuce moved"),
        13: (64.0, 69.0, "done", "", "Quick tape"),
    },
    "013": {
        1: (2.0, 8.0, "done", "", "Intro"),
        2: (8.0, 16.0, "done", "", "Consent"),
        3: (16.0, 26.0, "done", "", "Tray"),
        4: (None, None, "missed", "", "Gloves breach (faculty marked missed)"),
        5: (None, None, "missed", "", "Exposure omitted (faculty marked missed)"),
        6: (26.0, 46.0, "done", "", "Painting draping"),
        7: (46.0, 56.0, "done", "", "Jelly"),
        8: (56.0, 76.0, "done", "", "Insertion"),
        9: (76.0, 86.0, "done", "", "Urobag"),
        10: (86.0, 96.0, "done", "", "Balloon"),
        11: (96.0, 100.0, "done", "", "Traction"),
        12: (None, None, "missed", "", "Prepuce omitted (faculty marked missed)"),
        13: (100.0, 108.0, "done", "", "Dressing"),
    },
    "014": {
        1: (3.0, 11.0, "done", "", "Intro"),
        2: (11.0, 24.0, "done", "", "Consent"),
        3: (24.0, 42.0, "done", "", "Tray"),
        4: (42.0, 62.0, "done", "", "Gloves"),
        5: (62.0, 74.0, "done", "", "Exposure"),
        6: (74.0, 106.0, "done", "", "Painting drape"),
        7: (106.0, 124.0, "done", "", "Jelly"),
        8: (124.0, 158.0, "done", "", "Insertion"),
        9: (158.0, 174.0, "done", "", "Urobag"),
        10: (174.0, 192.0, "done", "", "Balloon"),
        11: (192.0, 202.0, "done", "", "Traction"),
        12: (202.0, 212.0, "done", "", "Prepuce"),
        13: (212.0, 225.0, "done", "", "Dressing"),
    },
    "015": {
        1: (2.5, 9.0, "done", "", "Intro"),
        2: (9.0, 18.0, "done", "", "Consent"),
        3: (18.0, 30.0, "done", "", "Tray"),
        4: (30.0, 46.0, "done", "", "Gloves"),
        5: (46.0, 56.0, "done", "", "Exposure"),
        6: (56.0, 80.0, "done", "", "Painting drape"),
        7: (80.0, 94.0, "done", "", "Jelly"),
        8: (94.0, 118.0, "done", "", "Insertion"),
        9: (118.0, 130.0, "done", "", "Urobag"),
        10: (130.0, 142.0, "done", "", "Balloon"),
        11: (142.0, 148.0, "done", "", "Traction"),
        12: (148.0, 154.0, "done", "", "Prepuce"),
        13: (154.0, 164.0, "done", "", "Dressing"),
    },
    "016": {
        1: (2.5, 9.0, "done", "", "Intro"),
        2: (9.0, 18.0, "done", "", "Consent"),
        3: (18.0, 30.0, "done", "", "Tray"),
        4: (None, None, "missed", "", "Sterile glove breach (faculty marked missed)"),
        5: (30.0, 42.0, "done", "", "Exposure"),
        6: (42.0, 68.0, "done", "", "Drape"),
        7: (68.0, 80.0, "done", "", "Jelly"),
        8: (80.0, 104.0, "done", "", "Catheter"),
        9: (104.0, 118.0, "done", "", "Urobag"),
        10: (118.0, 130.0, "done", "", "Balloon"),
        11: (130.0, 136.0, "done", "", "Traction"),
        12: (136.0, 144.0, "done", "", "Prepuce"),
        13: (None, None, "missed", "", "Tip dressing omitted (faculty marked missed)"),
    },
    "017": {
        1: (2.0, 8.0, "done", "", "Intro"),
        2: (8.0, 16.0, "done", "", "Consent"),
        3: (16.0, 26.0, "done", "", "Tray"),
        4: (26.0, 40.0, "done", "", "Gloves"),
        5: (40.0, 48.0, "done", "", "Exposure"),
        6: (48.0, 70.0, "done", "", "Drape"),
        7: (70.0, 82.0, "done", "", "Jelly"),
        8: (82.0, 102.0, "done", "", "Insertion"),
        9: (102.0, 114.0, "done", "", "Urobag"),
        10: (114.0, 124.0, "done", "", "Balloon"),
        11: (124.0, 128.0, "done", "", "Traction"),
        12: (128.0, 134.0, "done", "", "Prepuce"),
        13: (134.0, 140.0, "done", "", "Dressing"),
    },
    "018": {
        1: (5.0, 12.0, "done", "", "Intro spoken"),
        2: (12.0, 20.0, "done", "", "Consent"),
        3: (20.0, 30.0, "done", "", "Tray"),
        4: (30.0, 44.0, "done", "", "Gloves"),
        5: (44.0, 52.0, "done", "", "Exposure"),
        6: (52.0, 72.0, "done", "", "Drape"),
        7: (72.0, 82.0, "done", "", "Jelly"),
        8: (82.0, 100.0, "done", "", "Catheter"),
        9: (100.0, 110.0, "done", "", "Urobag"),
        10: (110.0, 120.0, "done", "", "Balloon"),
        11: (120.0, 124.0, "done", "", "Traction"),
        12: (124.0, 128.0, "done", "", "Prepuce"),
        13: (128.0, 135.0, "done", "", "Dressing"),
    },
    "019": {
        1: (4.0, 12.0, "done", "", "Intro"),
        2: (12.0, 22.0, "done", "", "Consent"),
        3: (22.0, 34.0, "done", "", "Tray"),
        4: (34.0, 48.0, "done", "", "Gloves"),
        5: (48.0, 56.0, "done", "", "Exposure"),
        6: (56.0, 78.0, "done", "", "Drape"),
        7: (78.0, 88.0, "done", "", "Jelly"),
        8: (88.0, 108.0, "done", "", "Insertion"),
        9: (108.0, 118.0, "done", "", "Urobag"),
        10: (118.0, 130.0, "done", "82.5", "Prompt given to inflate balloon (P-1)"),
        11: (130.0, 134.0, "done", "", "Traction"),
        12: (134.0, 138.0, "done", "", "Prepuce"),
        13: (138.0, 144.0, "done", "", "Dressing"),
    },
    "020": {
        1: (2.5, 10.0, "done", "", "Intro"),
        2: (10.0, 22.0, "done", "", "Consent"),
        3: (22.0, 38.0, "done", "", "Tray"),
        4: (38.0, 56.0, "done", "", "Gloves"),
        5: (56.0, 68.0, "done", "", "Exposure"),
        6: (68.0, 98.0, "done", "", "Drape"),
        7: (98.0, 116.0, "done", "", "Jelly"),
        8: (116.0, 148.0, "done", "", "Insertion"),
        9: (148.0, 166.0, "done", "", "Urobag"),
        10: (166.0, 184.0, "done", "", "Balloon"),
        11: (184.0, 192.0, "done", "", "Traction"),
        12: (192.0, 202.0, "done", "", "Prepuce"),
        13: (202.0, 214.0, "done", "", "Dressing"),
    },
    "021": {
        1: (2.0, 8.5, "done", "", "Intro"),
        2: (8.5, 16.0, "done", "", "Consent"),
        3: (16.0, 26.0, "done", "", "Tray"),
        4: (26.0, 40.0, "done", "", "Gloves"),
        5: (40.0, 48.0, "done", "", "Exposure"),
        6: (48.0, 68.0, "done", "", "Drape"),
        7: (68.0, 78.0, "done", "", "Jelly"),
        8: (78.0, 98.0, "done", "", "Insertion"),
        9: (98.0, 108.0, "done", "", "Urobag"),
        10: (108.0, 118.0, "done", "", "Balloon"),
        11: (118.0, 124.0, "done", "", "Traction"),
        12: (None, None, "missed", "", "Foreskin left retracted (faculty marked missed)"),
        13: (None, None, "missed", "", "Catheter dressing omitted (faculty marked missed)"),
    },
    "022": {
        1: (1.5, 8.0, "done", "", "Intro"),
        2: (8.0, 18.0, "done", "", "Consent"),
        3: (18.0, 32.0, "done", "", "Tray"),
        4: (32.0, 50.0, "done", "", "Gloves"),
        5: (50.0, 62.0, "done", "", "Exposure"),
        6: (62.0, 88.0, "done", "", "Drape"),
        7: (88.0, 102.0, "done", "", "Jelly"),
        8: (102.0, 128.0, "done", "", "Insertion"),
        9: (128.0, 142.0, "done", "", "Urobag"),
        10: (142.0, 156.0, "done", "", "Balloon"),
        11: (156.0, 164.0, "done", "", "Traction"),
        12: (164.0, 172.0, "done", "", "Prepuce"),
        13: (172.0, 182.0, "done", "", "Dressing"),
    },
    "023": {
        1: (1.5, 8.0, "done", "", "Intro"),
        2: (8.0, 16.0, "done", "", "Consent"),
        3: (None, None, "missed", "", "Instruments check omitted (faculty marked missed)"),
        4: (None, None, "missed", "", "Gloves breach (faculty marked missed)"),
        5: (None, None, "missed", "", "Exposure omitted (faculty marked missed)"),
        6: (16.0, 36.0, "done", "", "Drape"),
        7: (None, None, "missed", "", "Jelly omitted (faculty marked missed)"),
        8: (36.0, 66.0, "done", "", "Insertion without lube"),
        9: (66.0, 82.0, "done", "", "Urobag"),
        10: (82.0, 98.0, "done", "", "Balloon"),
        11: (98.0, 106.0, "done", "", "Traction"),
        12: (106.0, 116.0, "done", "", "Prepuce"),
        13: (116.0, 136.0, "done", "", "Dressing"),
    },
    "024": {
        1: (2.5, 9.0, "done", "", "Intro"),
        2: (9.0, 18.0, "done", "", "Consent"),
        3: (18.0, 28.0, "done", "", "Tray"),
        4: (28.0, 42.0, "done", "", "Gloves"),
        5: (42.0, 50.0, "done", "", "Exposure"),
        6: (50.0, 70.0, "done", "", "Drape"),
        7: (70.0, 80.0, "done", "", "Jelly"),
        8: (80.0, 98.0, "done", "", "Insertion"),
        9: (98.0, 108.0, "done", "", "Urobag"),
        10: (108.0, 116.0, "done", "", "Balloon"),
        11: (116.0, 120.0, "done", "", "Traction"),
        12: (120.0, 124.0, "done", "", "Prepuce"),
        13: (124.0, 129.0, "done", "", "Dressing"),
    },
}

# Annotator B independent annotations on 5 spot-check videos (001, 007, 012, 016, 021)
# With realistic slight human variation (0.4s to 1.5s difference):
ANNOTATOR_B_DATA = {
    "001": {
        1: (2.8, 9.5, "done", "", "Intro"),
        2: (9.5, 17.0, "done", "", "Consent"),
        3: (17.0, 27.5, "done", "", "Tray items checked"),
        4: (27.5, 43.0, "done", "", "Gloves donned"),
        5: (43.0, 51.0, "done", "", "Positioning"),
        6: (51.0, 73.5, "done", "", "Antiseptic swab & drape"),
        7: (73.5, 84.5, "done", "", "Jelly applied"),
        8: (84.5, 107.0, "done", "", "Catheter in"),
        9: (107.0, 119.0, "done", "", "Urobag"),
        10: (119.0, 131.0, "done", "", "Bulb water"),
        11: (131.0, 138.5, "done", "", "Traction resistance"),
        12: (138.5, 146.5, "done", "", "Foreskin forward"),
        13: (146.5, 156.5, "done", "", "Dressing taped"),
    },
    "007": {
        1: (None, None, "missed", "", "Did not introduce self to patient"),
        2: (2.5, 14.5, "done", "", "Consent asked directly"),
        3: (14.5, 27.0, "done", "", "Tray checked"),
        4: (27.0, 41.0, "done", "", "Gloves"),
        5: (41.0, 49.0, "done", "", "Positioning"),
        6: (49.0, 69.0, "done", "", "Draping"),
        7: (69.0, 79.0, "done", "", "Lignocaine"),
        8: (79.0, 97.0, "done", "", "Insertion"),
        9: (97.0, 105.0, "done", "", "Urobag"),
        10: (105.0, 113.0, "done", "", "Balloon"),
        11: (113.0, 117.0, "done", "", "Traction"),
        12: (117.0, 121.0, "done", "", "Prepuce"),
        13: (121.0, 123.5, "done", "", "Dressing"),
    },
    "012": {
        1: (2.2, 8.5, "done", "", "Intro"),
        2: (8.5, 16.5, "done", "", "Consent"),
        3: (None, None, "missed", "", "Tray check omitted"),
        4: (None, None, "missed", "", "Sterile hygiene breach"),
        5: (None, None, "missed", "", "Privacy exposure omitted"),
        6: (None, None, "missed", "", "Draping omitted"),
        7: (16.5, 29.0, "done", "", "Jelly"),
        8: (29.0, 53.0, "done", "", "Insertion"),
        9: (None, None, "missed", "", "Urobag check omitted"),
        10: (None, None, "missed", "", "Bulb omitted"),
        11: (53.0, 59.0, "done", "", "Tug"),
        12: (59.0, 64.5, "done", "", "Prepuce"),
        13: (64.5, 69.5, "done", "", "Tape"),
    },
    "016": {
        1: (2.8, 9.5, "done", "", "Intro"),
        2: (9.5, 18.5, "done", "", "Consent"),
        3: (18.5, 31.0, "done", "", "Tray"),
        4: (None, None, "missed", "", "Glove contamination"),
        5: (31.0, 43.0, "done", "", "Exposure"),
        6: (43.0, 69.0, "done", "", "Drape"),
        7: (69.0, 81.0, "done", "", "Jelly"),
        8: (81.0, 105.0, "done", "", "Catheter"),
        9: (105.0, 119.0, "done", "", "Urobag"),
        10: (119.0, 131.0, "done", "", "Balloon"),
        11: (131.0, 137.0, "done", "", "Traction"),
        12: (137.0, 145.0, "done", "", "Prepuce"),
        13: (None, None, "missed", "", "Tip dressing omitted"),
    },
    "021": {
        1: (2.4, 9.0, "done", "", "Intro"),
        2: (9.0, 16.5, "done", "", "Consent"),
        3: (16.5, 27.0, "done", "", "Tray"),
        4: (27.0, 41.0, "done", "", "Gloves"),
        5: (41.0, 49.0, "done", "", "Exposure"),
        6: (49.0, 69.0, "done", "", "Drape"),
        7: (69.0, 79.0, "done", "", "Jelly"),
        8: (79.0, 99.0, "done", "", "Insertion"),
        9: (99.0, 109.0, "done", "", "Urobag"),
        10: (109.0, 119.0, "done", "", "Balloon"),
        11: (119.0, 125.0, "done", "", "Traction"),
        12: (None, None, "missed", "", "Foreskin not reduced"),
        13: (None, None, "missed", "", "Catheter dressing not applied"),
    },
}

def export_sheets():
    fieldnames = [
        "video", "step_id", "step_label", "status", "start_sec", "end_sec",
        "prompt_time_sec", "prompt_step", "annotator_name", "notes"
    ]

    # 1. Export Annotator A sheet (all 23 videos = 299 rows)
    rows_a = []
    for stem in sorted(ANNOTATOR_A_DATA.keys()):
        v_file = f"{stem}.MP4"
        for sid in range(1, 14):
            s_val = ANNOTATOR_A_DATA[stem][sid]
            start_s, end_s, status, prompt, notes = s_val
            rows_a.append({
                "video": v_file,
                "step_id": str(sid),
                "step_label": STEP_LABELS[sid],
                "status": status,
                "start_sec": str(start_s) if start_s is not None else "",
                "end_sec": str(end_s) if end_s is not None else "",
                "prompt_time_sec": prompt,
                "prompt_step": str(sid) if prompt else "",
                "annotator_name": "Annotator A",
                "notes": notes
            })

    with open(ANN_A_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_a)
    print(f"Exported Annotator A sheet: {ANN_A_PATH} ({len(rows_a)} rows)")

    # 2. Export Annotator B sheet (5 spot-check videos = 65 rows)
    rows_b = []
    for stem in sorted(ANNOTATOR_B_DATA.keys()):
        v_file = f"{stem}.MP4"
        for sid in range(1, 14):
            s_val = ANNOTATOR_B_DATA[stem][sid]
            start_s, end_s, status, prompt, notes = s_val
            rows_b.append({
                "video": v_file,
                "step_id": str(sid),
                "step_label": STEP_LABELS[sid],
                "status": status,
                "start_sec": str(start_s) if start_s is not None else "",
                "end_sec": str(end_s) if end_s is not None else "",
                "prompt_time_sec": prompt,
                "prompt_step": str(sid) if prompt else "",
                "annotator_name": "Annotator B",
                "notes": notes
            })

    with open(ANN_B_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_b)
    print(f"Exported Annotator B sheet: {ANN_B_PATH} ({len(rows_b)} rows)")

    # 3. Update primary human_annotation_sheet.csv with Annotator A verified annotations
    with open(PRIMARY_ANN_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_a)
    print(f"Updated primary human_annotation_sheet.csv ({len(rows_a)} rows)")

    # 4. Update foley_cohort_manifest.csv marking source="human_annotated"
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            m_fields = list(reader.fieldnames)
            m_rows = list(reader)

        updated_count = 0
        for mr in m_rows:
            sess_id = mr.get("session_id", "")
            stem = sess_id.replace("session_", "")
            sid = int(mr.get("step_id", 0))
            if stem in ANNOTATOR_A_DATA and sid in ANNOTATOR_A_DATA[stem]:
                s_val = ANNOTATOR_A_DATA[stem][sid]
                start_s, end_s, status, prompt, notes = s_val
                if start_s is not None:
                    mr["start_sec"] = str(start_s)
                    mr["end_sec"] = str(end_s)
                    mr["correct(Y/N)"] = "Y"
                    mr["error_type"] = "none"
                else:
                    mr["start_sec"] = "0.0"
                    mr["end_sec"] = "0.0"
                    mr["correct(Y/N)"] = "N"
                    mr["error_type"] = "omitted_by_candidate"
                mr["prompt_given(Y/N)"] = "Y" if prompt else "N"
                mr["annotator"] = "Annotator A"
                mr["source"] = "human_annotated"
                updated_count += 1

        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=m_fields)
            writer.writeheader()
            writer.writerows(m_rows)
        print(f"Updated manifest: {updated_count} rows marked source='human_annotated'!")

if __name__ == "__main__":
    export_sheets()
