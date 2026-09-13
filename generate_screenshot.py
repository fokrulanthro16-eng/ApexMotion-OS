"""
Automated Live HUD Capture Utility for ApexMotion OS
Captures 1920x1080 high-resolution mission control state to docs/images/dashboard.png
"""

import asyncio
import os
from playwright.async_api import async_playwright

async def capture_dashboard(output_path="docs/images/dashboard.png", url="http://localhost:8000"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        print(f"Navigating to {url}...")
        await page.goto(url, wait_until="networkidle", timeout=15000)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=output_path, full_page=False)
        await browser.close()
        print(f"Dashboard screenshot captured to {output_path}")

if __name__ == "__main__":
    asyncio.run(capture_dashboard())
