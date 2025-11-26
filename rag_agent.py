# backend/rag_agent.py
import json
import re
from typing import List, Dict, Any
from .transformer_model import LocalHFModel
from .utils import safe_json_parse

class RAGAgent:
    def __init__(self, vectorstore=None):
        """
        vectorstore is optional but kept for compatibility.
        """
        self.vectorstore = vectorstore
        self.model = LocalHFModel()  # local HF model; may still fail but we handle it

    # -----------------------------
    # High-level public methods
    # -----------------------------
    def generate_test_cases(self, query: str) -> Dict[str, Any]:
        """
        Try model generation -> parse JSON -> if invalid, return deterministic fallback.
        Returns dict containing "testcases": [...]
        """
        # Build a guarded prompt that asks for JSON
        prompt = self._build_testcase_prompt(query)
        raw = ""
        try:
            raw = self.model.generate(prompt, max_tokens=700)
        except Exception as e:
            # Model error: log raw and fall back
            raw = ""

        # Try to parse model output
        parsed = safe_json_parse(raw)
        if parsed and isinstance(parsed, dict) and "testcases" in parsed:
            # sanity check: ensure non-empty testcases
            tcs = parsed.get("testcases") or []
            if isinstance(tcs, list) and len(tcs) >= 1 and self._tc_list_valid(tcs):
                return parsed

        # If we reach here, model output was invalid/empty -> fallback
        fallback = self._deterministic_testcase_generator(query)
        return {"testcases": fallback}

    def generate_selenium_script(self, testcase: Dict[str, Any]) -> str:
        """
        Try to generate using model; if model output is not code or is missing,
        return deterministic script based on testcase fields.
        """
        prompt = self._build_script_prompt(testcase)
        try:
            code_raw = self.model.generate(prompt, max_tokens=600)
        except Exception:
            code_raw = ""

        # naive detection whether model returned Python code (presence of 'import' or 'webdriver')
        if code_raw and ("import" in code_raw or "webdriver" in code_raw or "def " in code_raw):
            return code_raw

        # Fallback deterministic script:
        return self._deterministic_script_generator(testcase)

    # -----------------------------
    # Prompt builders
    # -----------------------------
    def _build_testcase_prompt(self, query: str) -> str:
        return f"""
You are a careful QA engineer. Given the requirement below, return ONLY valid JSON with the exact structure:
{{ "testcases": [ {{ "Test_ID": "", "Title": "", "Objective": "", "Preconditions": [], "Steps": [], "Expected_Result": "" }} ] }}

Produce 6-10 test cases (positive, negative, edge). Do NOT include explanation text outside JSON.

Requirement:
{query}
"""

    def _build_script_prompt(self, testcase: Dict[str, Any]) -> str:
        return f"""
You are an expert Selenium (Python) engineer. Generate a runnable Python Selenium script (Chrome) that implements the following test case.
Return only Python code, no explanation.

Test case:
{json.dumps(testcase, indent=2)}
"""

    # -----------------------------
    # Deterministic fallback generators
    # -----------------------------
    def _deterministic_testcase_generator(self, query: str) -> List[Dict[str, Any]]:
        """
        Create a list of test cases based on keywords in the query.
        This always returns valid, populated test cases.
        """
        q = query.lower()

        # Basic templates
        generic_positive = {
            "Test_ID": "TC_POS_001",
            "Title": "Apply valid discount code",
            "Objective": "Verify the checkout accepts a valid discount code and updates total.",
            "Preconditions": ["User on checkout page with items in cart"],
            "Steps": [
                "Open checkout page",
                "Enter a valid discount code (e.g., SAVE15) in the discount field",
                "Click the Apply button"
            ],
            "Expected_Result": "Discount applied and total updated to reflect discount."
        }

        generic_positive2 = {
            "Test_ID": "TC_POS_002",
            "Title": "Apply discount code with uppercase letters",
            "Objective": "Verify the system treats uppercase or lowercase discount codes equivalently when appropriate.",
            "Preconditions": ["Checkout page is loaded"],
            "Steps": [
                "Enter 'SAVE15' (uppercase) in the discount field",
                "Click Apply"
            ],
            "Expected_Result": "Discount applied successfully."
        }

        generic_negative = {
            "Test_ID": "TC_NEG_001",
            "Title": "Apply invalid discount code",
            "Objective": "Verify invalid or malformed codes are rejected.",
            "Preconditions": ["Checkout page is loaded"],
            "Steps": [
                "Enter an invalid discount code 'INVALID123'",
                "Click Apply"
            ],
            "Expected_Result": "Show 'Invalid code' error and do not change total."
        }

        generic_negative2 = {
            "Test_ID": "TC_NEG_002",
            "Title": "Apply empty discount code",
            "Objective": "Verify empty input is handled with validation.",
            "Preconditions": ["Checkout page is loaded"],
            "Steps": [
                "Leave discount field empty",
                "Click Apply"
            ],
            "Expected_Result": "Show 'Enter a code' or similar validation message."
        }

        generic_edge = {
            "Test_ID": "TC_EDGE_001",
            "Title": "Apply discount code twice",
            "Objective": "Verify same code cannot be stacked/applied twice on same cart.",
            "Preconditions": ["Discount code previously applied"],
            "Steps": [
                "Apply a valid code",
                "Attempt to apply the same code again"
            ],
            "Expected_Result": "Show 'Code already used' or prevent further discount application."
        }

        shipping_edge = {
            "Test_ID": "TC_EDGE_002",
            "Title": "Discount with shipping selection",
            "Objective": "Verify discount calculation with different shipping methods.",
            "Preconditions": ["Cart with items"],
            "Steps": [
                "Select Express shipping",
                "Apply valid discount code",
                "Verify total includes shipping and discount properly"
            ],
            "Expected_Result": "Total equals subtotal + shipping - discount (if discount applies to subtotal only)."
        }

        # If query mentions "discount" produce discount-specific list, otherwise create generic tests
        cases = []
        if "discount" in q or "coupon" in q or "promo" in q:
            cases = [
                generic_positive,
                generic_positive2,
                generic_negative,
                generic_negative2,
                generic_edge,
                shipping_edge
            ]
        else:
            # generic set for other features
            cases = [
                {
                    "Test_ID": "TC_POS_001",
                    "Title": "Positive flow - basic functionality",
                    "Objective": "Verify primary happy path works",
                    "Preconditions": ["User logged in if required"],
                    "Steps": ["Perform primary action", "Verify success indicator"],
                    "Expected_Result": "Primary function completes successfully"
                },
                {
                    "Test_ID": "TC_NEG_001",
                    "Title": "Negative flow - invalid input",
                    "Objective": "Verify invalid input is handled",
                    "Preconditions": ["Feature available"],
                    "Steps": ["Enter invalid input", "Submit"],
                    "Expected_Result": "Error message shown and no action taken"
                },
                {
                    "Test_ID": "TC_EDGE_001",
                    "Title": "Edge case - large input",
                    "Objective": "Verify system handles large inputs",
                    "Preconditions": ["Feature available"],
                    "Steps": ["Input very large value", "Submit"],
                    "Expected_Result": "Handled gracefully or validation error shown"
                },
                {
                    "Test_ID": "TC_EDGE_002",
                    "Title": "Concurrency/Repeat action",
                    "Objective": "Verify repeated actions do not corrupt state",
                    "Preconditions": ["Feature accessible"],
                    "Steps": ["Perform action multiple times quickly"],
                    "Expected_Result": "No corrupted state; idempotent behavior if required"
                },
                {
                    "Test_ID": "TC_POS_002",
                    "Title": "Boundary value test",
                    "Objective": "Verify behavior at boundary values",
                    "Preconditions": ["Feature available"],
                    "Steps": ["Enter boundary value", "Submit"],
                    "Expected_Result": "Correct behavior at boundary condition"
                },
                {
                    "Test_ID": "TC_NEG_002",
                    "Title": "Missing required field",
                    "Objective": "Verify required field validation",
                    "Preconditions": ["Feature available"],
                    "Steps": ["Omit a required field", "Submit"],
                    "Expected_Result": "Validation message and no acceptance"
                }
            ]

        # Ensure test IDs are unique and sequential when fallback is used multiple times:
        for i, tc in enumerate(cases, start=1):
            # If Test_ID missing or generic, normalize to TCxxx
            tc_id = f"TC_{i:03d}"
            tc["Test_ID"] = tc_id

            # Fill any empty strings if present
            for k in ["Title", "Objective", "Expected_Result"]:
                if k in tc and (not tc[k] or str(tc[k]).strip() == ""):
                    tc[k] = f"{tc.get('Title','No Title')} - {k} auto-filled"

            # Ensure Steps and Preconditions are lists of non-empty strings
            tc["Steps"] = [s for s in tc.get("Steps", []) if str(s).strip()]
            tc["Preconditions"] = [p for p in tc.get("Preconditions", []) if str(p).strip()]

        return cases

    def _deterministic_script_generator(self, testcase: Dict[str, Any]) -> str:
        """
        Build a simple Selenium script implementing the steps in the testcase.
        This is deterministic and will run if the selectors are adjusted to the real page.
        """
        title_safe = re.sub(r'[^0-9A-Za-z_]+', '_', testcase.get("Test_ID", "TC"))
        # Build a minimal script
        steps = testcase.get("Steps", [])
        script_lines = [
            "from selenium import webdriver",
            "from selenium.webdriver.common.by import By",
            "import time",
            "",
            "driver = webdriver.Chrome()",
            "try:",
            "    driver.maximize_window()",
            "    # TODO: replace with your local path or server URL",
            "    driver.get('file:///PATH/TO/checkout.html')",
            ""
        ]

        # naive mapping of common step phrases to DOM actions
        for s in steps:
            s_lower = s.lower()
            if "enter" in s_lower and "discount" in s_lower:
                script_lines += [
                    "    # Enter discount code (adjust selector if needed)",
                    "    el = driver.find_element(By.ID, 'discount-code')",
                    "    el.clear()",
                    "    el.send_keys('SAVE15')",
                    "    time.sleep(0.5)"
                ]
            elif "click" in s_lower or "apply" in s_lower or "submit" in s_lower:
                script_lines += [
                    "    # Click apply/pay (adjust selector if needed)",
                    "    try:",
                    "        btn = driver.find_element(By.ID, 'apply-discount')",
                    "    except:",
                    "        btn = driver.find_element(By.CSS_SELECTOR, 'button')",
                    "    btn.click()",
                    "    time.sleep(1)"
                ]
            elif "select express" in s_lower or "shipping" in s_lower:
                script_lines += [
                    "    # Select shipping option (adjust selector if needed)",
                    "    try:",
                    "        driver.find_element(By.CSS_SELECTOR, \"input[name='shipping'][value='express']\").click()",
                    "    except Exception:",
                    "        pass",
                    "    time.sleep(0.5)"
                ]
            else:
                # generic step
                script_lines += [
                    f"    # Step: {s}",
                    "    time.sleep(0.5)"
                ]

        # Verification placeholder
        script_lines += [
            "",
            "    # Verification placeholder -- update selectors/assertions as needed",
            "    try:",
            "        success = driver.find_element(By.ID, 'payment-success')",
            "        if success.is_displayed():",
            "            print('PASS')",
            "        else:",
            "            print('FAIL')",
            "    except Exception:",
            "        print('VERIFY_MANUALLY')",
            "",
            "finally:",
            "    driver.quit()"
        ]

        return "\n".join(script_lines)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _tc_list_valid(self, tcs: List[Dict[str, Any]]) -> bool:
        """
        Basic validation: are required fields present and non-empty?
        """
        if not isinstance(tcs, list) or len(tcs) < 1:
            return False
        for tc in tcs:
            if not isinstance(tc, dict):
                return False
            # ensure at least Title and Steps or Expected_Result
            if not tc.get("Title") or not tc.get("Steps"):
                return False
        return True
