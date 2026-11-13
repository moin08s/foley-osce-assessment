"""OSCE Grading and Scorecard Generator.

D Y Patil Medical College - Procedural Skills Assessment Model
Male Foley's Catheterization OSCE Examination
Pure Python Standard Library + NumPy (no third-party dependencies).

Grading Scheme:
- "P"   : Pass (Advance Beginner) - All 13 steps performed correctly in order.
- "P-1" : Pass with 1 Prompt (Beginner) - Exactly 1 step corrected after verbal prompt.
- "R"   : Remediation (Novice) - Missing steps, critical safety errors, or >1 prompts.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .data_loader import STEP_LABELS


class OSCEGrader:
    """Grades a candidate's session and outputs both structured JSON and printable text scorecards."""

    def __init__(self, institution: str = "D Y Patil Medical College"):
        self.institution = institution
        self.procedure_name = "Male Foley's Catheterization"

    def compute_grade(
        self,
        sequence_check: Dict[str, Any],
        student_id: str,
        session_id: str,
        detected_segments: List[Dict[str, Any]],
        faculty_ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute final OSCE grade and compile assessment report.
        
        Rules:
        - "P": All 13 steps present, correct order, zero critical safety violations, 0 prompts.
        - "P-1": Exactly one step corrected after prompt, no uncorrected critical safety violations, no missing steps.
        - "R": Anything else (missing steps, critical safety violations, >1 prompts).
        """
        all_present = sequence_check["all_13_present"]
        violations = sequence_check["violations"]
        crit_violations = sequence_check["critical_safety_violations"]
        prompts_count = sequence_check["prompts_count"]
        missing_steps = sequence_check["missing_steps"]

        # Step pass/fail status dict
        step_status = {}
        for s in range(1, 14):
            lbl = STEP_LABELS[s]
            if s in missing_steps:
                step_status[s] = {"step_id": s, "label": lbl, "status": "OMITTED", "passed": False}
            else:
                # Find corresponding segment
                seg = next((x for x in detected_segments if x["step_id"] == s), None)
                is_prompted = s in sequence_check.get("prompts_given", [])
                status_str = "PROMPTED (P-1)" if is_prompted else "PERFORMED (Pass)"
                step_status[s] = {
                    "step_id": s,
                    "label": lbl,
                    "status": status_str,
                    "passed": not is_prompted,
                    "start_sec": seg["start_sec"] if seg else None,
                    "end_sec": seg["end_sec"] if seg else None,
                    "duration_sec": seg["duration_sec"] if seg else None,
                    "confidence": seg["confidence"] if seg else None,
                }

        # Total defects count: missing steps + prompts
        total_defects = len(missing_steps) + prompts_count

        # Grade determination
        if all_present and len(crit_violations) == 0 and len(violations) == 0 and total_defects == 0:
            final_grade = "P"
            grade_description = "Pass: Advance Beginner (Demonstrates independent competence; all 13 steps completed safely in sequence without prompting)"
        elif len(crit_violations) == 0 and total_defects == 1:
            final_grade = "P-1"
            grade_description = "Pass with 1 Prompt / Minor Defect: Beginner (Competent performance with one minor prompt/defect; zero critical safety compromise)"
        else:
            final_grade = "R"
            grade_description = "Needs Remediation: Novice (Procedural omissions, critical safety breaches, or multiple prompts require faculty remediation before unassisted clinical practice)"

        # Calculate numerical score out of 13
        steps_passed = sum(1 for s in step_status.values() if s["passed"])
        score_percent = round((steps_passed / 13.0) * 100.0, 1)

        # Faculty ground truth comparison
        faculty_alignment = None
        if faculty_ground_truth is not None:
            faculty_grade = faculty_ground_truth.get("final_grade", "P")
            match = (final_grade == faculty_grade)
            discrepancies = []
            if not match:
                discrepancies.append(f"Grade mismatch: Model predicted '{final_grade}', Faculty assigned '{faculty_grade}'.")

            faculty_steps = faculty_ground_truth.get("step_results", {})
            for sid_str, f_stat in faculty_steps.items():
                sid = int(sid_str)
                m_pass = step_status.get(sid, {}).get("passed", False)
                f_pass = (f_stat in ["tick", "Pass", "P", "Y"])
                if m_pass != f_pass:
                    discrepancies.append(f"Step {sid} ({STEP_LABELS.get(sid)}): Model={m_pass}, Faculty={f_stat}")

            faculty_alignment = {
                "faculty_grade": faculty_grade,
                "model_grade": final_grade,
                "grade_match": match,
                "discrepancies": discrepancies,
            }

        report = {
            "institution": self.institution,
            "procedure": self.procedure_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_id": student_id,
            "session_id": session_id,
            "final_grade": final_grade,
            "grade_description": grade_description,
            "score": f"{steps_passed}/13 ({score_percent}%)",
            "all_13_present": all_present,
            "prompts_count": prompts_count,
            "detected_segments": detected_segments,
            "step_results": step_status,
            "ordering_violations": violations,
            "critical_safety_violations": crit_violations,
            "faculty_alignment": faculty_alignment,
        }

        return report

    def format_text_report(self, report: Dict[str, Any]) -> str:
        """Format assessment report into a clean, printable ASCII OSCE evaluation sheet."""
        lines = []
        border = "=" * 80
        thin_border = "-" * 80

        lines.append(border)
        lines.append(f"      {report['institution'].upper()} - MEDICAL EDUCATION UNIT")
        lines.append(f"           OSCE CLINICAL SKILLS ASSESSMENT REPORT")
        lines.append(f"              Procedure: {report['procedure']}")
        lines.append(border)
        lines.append(f"Candidate ID : {report['student_id']:<20} Session ID : {report['session_id']}")
        lines.append(f"Evaluated At : {report['timestamp']}")
        lines.append(f"Final Score  : {report['score']:<20} FINAL GRADE: [ {report['final_grade']} ]")
        lines.append(f"Grade Meaning: {report['grade_description']}")
        lines.append(thin_border)
        lines.append("  # | PROCEDURAL STEP               | STATUS        | TIME (s)     | CONF")
        lines.append(thin_border)

        for sid in range(1, 14):
            step_info = report["step_results"][sid]
            name = step_info["label"]
            status = step_info["status"]
            if step_info.get("start_sec") is not None:
                timing = f"{step_info['start_sec']:>5.1f} - {step_info['end_sec']:>5.1f}s"
                conf = f"{step_info['confidence']*100:>4.1f}%"
            else:
                timing = "    --        "
                conf = "  -- "

            mark = "[x]" if step_info["passed"] else "[ ]"
            lines.append(f" {sid:>2} | {mark} {name:<25} | {status:<13} | {timing} | {conf}")

        lines.append(thin_border)
        lines.append("PROCEDURAL SEQUENCE & SAFETY AUDIT:")
        if len(report["critical_safety_violations"]) > 0:
            lines.append("  [!] CRITICAL SAFETY VIOLATIONS:")
            for csv in report["critical_safety_violations"]:
                lines.append(f"      - {csv}")
        else:
            lines.append("  [*] Zero critical safety violations detected.")

        if len(report["ordering_violations"]) > 0:
            lines.append("  [!] Ordering / Communication Flags:")
            for viol in report["ordering_violations"]:
                lines.append(f"      - [{viol.get('severity', 'FLAG')}] {viol.get('rule')}: {viol.get('description')}")
        else:
            lines.append("  [*] All steps performed in strict chronological order.")

        if report.get("faculty_alignment"):
            fa = report["faculty_alignment"]
            lines.append(thin_border)
            lines.append("FACULTY GROUND TRUTH VALIDATION (SAMPLE BENCHMARK):")
            lines.append(f"  Faculty Assigned Grade : {fa['faculty_grade']}")
            lines.append(f"  Model Predicted Grade  : {fa['model_grade']}")
            lines.append(f"  Grade Agreement        : {'PERFECT MATCH (PASS)' if fa['grade_match'] else 'DISCREPANCY'}")
            if fa["discrepancies"]:
                for disc in fa["discrepancies"]:
                    lines.append(f"      - {disc}")
            else:
                lines.append("  Discrepancies          : None (100% agreement with faculty ground truth)")

        lines.append(border)
        lines.append("  Faculty Signature: _______________________    Date: ______________")
        lines.append(border)
        return "\n".join(lines)
