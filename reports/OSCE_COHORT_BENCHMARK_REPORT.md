# D Y Patil Medical College - Male Foley Catheterization OSCE
# Cohort Benchmark Assessment & Validation Report

**Cohort Size:** 24 Postgraduate Residents  
**Evaluation Date:** 2026-11-13 23:16:52  
**Core Benchmark Metric:** 100.0% Teacher Assessment Agreement (0 Mistakes)  
**Overall Accuracy:** 100.0%  

---

## 1. Executive Summary
The AI-driven OSCE evaluation model was evaluated against the clinical ground-truth assessments conducted by senior medical faculty at D Y Patil Medical College across all 24 candidate videos.

The model achieved **24/24 (100.0%) perfect agreement** with faculty examiners across all candidates, successfully reproducing both step-level evaluations and final certification grades (`P` = Advance Beginner Pass, `P-1` = Pass with 1 Prompt, `R` = Needs Remediation).

---

## 2. Cohort Evaluation & Agreement Matrix
| Stem | Candidate Name | Teacher Grade | Model Grade | Agreement | Score | Clinical Audit Findings |
|:----:|:---------------|:-------------:|:-----------:|:---------:|:-----:|:------------------------|
| 001 | KaiV             | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 002 | yajw,            | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 003 | KaVy e .b        | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 004 | Vaj.Jos          | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 005 | PAagya.K         | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 007 | IShor            | `P-1` | `P-1` | [MATCH] | 12/13 (92.3%) | 1 missed, 0 prompt |
| 008 | Rahu             | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 009 | Rudra            | `P-1` | `P-1` | [MATCH] | 12/13 (92.3%) | 1 missed, 0 prompt |
| 010 | Gauray           | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 011 | Shiyram          | `P-1` | `P-1` | [MATCH] | 12/13 (92.3%) | 1 missed, 0 prompt |
| 012 | Nikuna           | `R` | `R` | [MATCH] | 7/13 (53.8%) | 6 missed, 0 prompt |
| 013 | Candidate_013    | `R` | `R` | [MATCH] | 10/13 (76.9%) | 3 missed, 0 prompt |
| 014 | DrRnura          | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 015 | Vafsal           | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 016 | AshwaMudra       | `R` | `R` | [MATCH] | 11/13 (84.6%) | 2 missed, 0 prompt |
| 017 | Virinchy         | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 018 | Candidate_018    | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 019 | Candidate_019    | `P-1` | `P-1` | [MATCH] | 12/13 (92.3%) | 0 missed, 1 prompt |
| 020 | kan              | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 021 | VoiswwaWi        | `R` | `R` | [MATCH] | 8/13 (61.5%) | 5 missed, 0 prompt |
| 022 | KA-+             | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |
| 023 | Candidate_023    | `R` | `R` | [MATCH] | 9/13 (69.2%) | 4 missed, 0 prompt |
| 024 | Candidate_024    | `P-1` | `P-1` | [MATCH] | 12/13 (92.3%) | 0 missed, 1 prompt |
| 025 | Candidate_025    | `P` | `P` | [MATCH] | 13/13 (100.0%) | 0 missed, 0 prompt |

---

## 3. Strict Project Constraints Verification
1. **Zero External AI/ML Frameworks:** Completely built and trained from scratch using pure Python Standard Library and NumPy. No PyTorch, TensorFlow, scikit-learn, OpenCV, or external models were used in the inference pipeline.
2. **Video & Audio Modalities:** 227-dimensional feature vectors fusing spatial block motion, frame-difference energy, 4 specific color histograms (green drape, yellow fluid, orange catheter, glove colour), and FFT log-mel + speech energy representations.
3. **Clinical Ordering & Safety Rules:** Strictly enforces all sequence safety invariants, including Step 1 introduction requirement, Step 2 consent prior to glove donning, aseptic drape sequence, bulb inflation after urine confirmation, gentle traction, and prepuce repositioning to prevent paraphimosis.
