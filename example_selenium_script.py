# examples/example_selenium_script.py
"""
Generated Selenium script example for test case: Apply valid discount code 'SAVE15'
This example uses Chrome WebDriver. Make sure chromedriver is in PATH or provide executable_path.
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import math

driver = webdriver.Chrome()
try:
    driver.get("C:\Users\surya\Desktop\Assignment-1\checkout.html")  # <-- replace with local path

    # Add items to cart (Item A and Item B)
    driver.find_element(By.CSS_SELECTOR, "button.add-to-cart[data-id='itemA']").click()
    driver.find_element(By.CSS_SELECTOR, "button.add-to-cart[data-id='itemB']").click()
    time.sleep(0.5)

    # Open quantity inputs and set quantities
    qty_inputs = driver.find_elements(By.CSS_SELECTOR, "#cart-items input.qty")
    # set first two to 1 (default). No change needed here.

    # Apply discount code SAVE15
    discount_input = driver.find_element(By.ID, "discount-code")
    discount_input.clear()
    discount_input.send_keys("SAVE15")
    driver.find_element(By.ID, "apply-discount").click()
    time.sleep(0.5)

    # Check total is reduced by 15%
    total_elem = driver.find_element(By.ID, "total")
    total_text = total_elem.text
    total_val = float(total_text)
    # Compute expected: subtotal = 30 + 20 = 50, shipping = 0, expected = 50*0.85 = 42.5
    expected = round((30 + 20) * 0.85, 2)
    assert abs(total_val - expected) < 0.01, f"Total {total_val} != expected {expected}"

    # Fill user details
    driver.find_element(By.ID, "name").send_keys("Test User")
    driver.find_element(By.ID, "email").send_keys("test@example.com")
    driver.find_element(By.ID, "address").send_keys("123 Main St")

    # Choose payment and click Pay Now
    driver.find_element(By.CSS_SELECTOR, "input[name='payment'][value='card']").click()
    driver.find_element(By.ID, "pay-now").click()
    time.sleep(0.5)

    # Verify Payment Successful message
    success = driver.find_element(By.ID, "payment-success")
    assert success.is_displayed(), "Payment success not displayed"

    print("Test passed: Discount applied and payment succeeded.")

finally:
    driver.quit()
