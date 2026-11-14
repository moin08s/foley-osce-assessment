# D Y PATIL MEDICAL COLLEGE & HOSPITAL
### Male Foley Catheterization OSCE - Verified Faculty Checklist
**Candidate:** Resident-016 (AshwaMudra) | **Faculty Ground-Truth Grade:** R

> [!IMPORTANT]
> **Audit & Evaluation Policy:**
> 1. **Detector Metrics (Reported Twice):**
>    - **Real Use (Without Contradiction Suppression):** Step 4 False-Alarm = 50.0% (2/4), Step 6 False-Alarm = 50.0% (1/2), Step 9 False-Alarm = 0.0% (0/1), Step 10 False-Alarm = 100.0% (1/1).
>    - **With Contradiction Suppression:** Apparent False-Alarm = 0.0% across all steps because the faculty sheet overrides false alarms.
> 2. **Reliability Definition:** 'High' requires testing on >= 5 negative sessions AND sensitivity >= 90%. Steps 2, 8, 11 have 'No negative data'. All other steps are 'Untested/Low'.

| Step | Clinical Step Name | AI Status | First Signal | Reliability | Measured Signal Evidence | Evidence Source | Faculty Confirmation |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **01** | introduce_explain | <span style='color:gray;'>NOT VERIFIED</span> | `—` | Untested/Low (1 neg sample) | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |
| **02** | informed_consent | <span style='color:gray;'>NOT VERIFIED</span> | `—` | No negative data | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |
| **03** | check_instruments | <span style='color:gray;'>NOT VERIFIED</span> | `—` | Untested/Low (2 neg samples) | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |
| **04** | hand_hygiene_gloves | <span style='color:red;'>**KNOWN FAILURE / FALSE ALARM (Suppressed)**</span> | `00:00` | Untested/Low (4 neg samples, 50.0% False-Alarm Rate) | Signal detected (5.7% peak white reflection), but contradicted by faculty ground truth (Resident missed step) | Glove Colour Detector (Contradicted) | [ ] Confirm<br>[ ] Overrule |
| **05** | privacy_position_exposure | <span style='color:gray;'>NOT VERIFIED</span> | `—` | Untested/Low (3 neg samples) | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |
| **06** | painting_draping | **FOUND (✔)** | `00:00` | Untested/Low (2 neg samples, 50.0% False-Alarm Rate) | 5.9 per-mille green drape coverage | Green Drape Detector | [ ] Confirm<br>[ ] Overrule |
| **07** | lignocaine_jelly | <span style='color:gray;'>NOT VERIFIED</span> | `—` | Untested/Low (1 neg sample) | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |
| **08** | open_catheter_insert | <span style='color:gray;'>NOT VERIFIED</span> | `—` | No negative data | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |
| **09** | attach_urobag_confirm_urine | **FOUND (✔)** | `01:06` | Untested/Low (1 neg sample, 54.5% Sensitivity) | 12.2 per-mille yellow urine flash | Urine Yellow Detector | [ ] Confirm<br>[ ] Overrule |
| **10** | inflate_balloon | **FOUND (✔)** | `01:10` | Untested/Low (1 neg sample, 100.0% False-Alarm Rate) | 16.1% pelvic motion burst | Pelvic Motion Detector (Syringe Proxy) | [ ] Confirm<br>[ ] Overrule |
| **11** | traction_until_resistance | <span style='color:gray;'>NOT VERIFIED</span> | `—` | No negative data | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |
| **12** | reposition_prepuce | <span style='color:gray;'>NOT VERIFIED</span> | `—` | Untested/Low (2 neg samples) | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |
| **13** | tip_dressing | <span style='color:gray;'>NOT VERIFIED</span> | `—` | Untested/Low (3 neg samples) | No detector signal | Uninstrumented | [ ] Confirm<br>[ ] Overrule |