#!/usr/bin/env python3
# deeptmhmm_batch_submit.py

import os
import re
import sys
import time
import shutil
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------- CONFIG ----------------------
DTU_URL = "https://services.healthtech.dtu.dk/services/DeepTMHMM-1.0/"

BASE_DIR = Path(sys.argv[1])

CHUNK_DIR  = BASE_DIR / "deeptmhmm_chunks"
RESULT_DIR = BASE_DIR / "deeptmhmm_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


PAGE_LOAD_TIMEOUT   = 120
RESULT_WAIT_TIMEOUT = 900

# ---------------------- DRIVER ----------------------
def build_driver(download_dir: str):
    download_dir = str(Path(download_dir).resolve())
    os.makedirs(download_dir, exist_ok=True)

    opts = Options()
    opts.add_argument("--headless=new")  # comment to watch the browser
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1400,2000")
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver

# ---------------------- PARSERS ----------------------
def extract_3line(page_text: str) -> str:
    """
    Extract the 'Predicted Topologies' block (3-line format):
      >id | TYPE
      SEQUENCE
      TOPOLOGY-LINE (SSS/OOO/III etc)
    """
    # Grab the section starting at "Predicted Topologies"
    m = re.search(r"Predicted Topologies\s*(.+?)(?:\n{2,}|\Z)", page_text, flags=re.DOTALL)
    if not m:
        return ""
    block = m.group(1).strip()

    # Keep only lines that are part of the 3line block (>, letters, topology letters)
    lines = []
    for line in block.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith('>'):
            lines.append(line)
            continue
        # sequence line (A-Z only) or topology line (S/O/I/M chars)
        if re.fullmatch(r"[A-Z]+", line) or re.fullmatch(r"[SOIM]+", line):
            lines.append(line)

    # Validate triplets (>header, seq, topology). If not aligned, we still dump as-is.
    return "\n".join(lines) + ("\n" if lines else "")

def extract_gff3(page_text: str) -> str:
    """
    Extract everything from '##gff-version 3' until the end of the gff block
    (we include the trailing '//' separators and the 'Job summary' header as a natural stop).
    """
    start = page_text.find("##gff-version 3")
    if start == -1:
        return ""
    tail = page_text[start:]
    # Stop before "Job summary" if present; else take to end
    end_header = re.search(r"\n\s*Job summary", tail)
    if end_header:
        tail = tail[:end_header.start()]
    return tail.strip() + ("\n" if tail.strip() else "")

def save_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

# ---------------------- CORE ----------------------
def submit_one_chunk(driver, fasta_path: Path, download_root: Path):
    chunk_name = fasta_path.stem  # "chunk_0001"
    print(f"[+] Submitting {fasta_path.name}")
    driver.get(DTU_URL)

    # 1) File input
    file_selectors = [
        (By.CSS_SELECTOR, "input[type='file']"),
        (By.NAME, "SEQSUB_file"),
        (By.ID, "file"),
        (By.CSS_SELECTOR, "input.form-control-file[type='file']"),
    ]

    def first_present(selectors, timeout=40):
        end = time.time() + timeout
        last_err = None
        while time.time() < end:
            for by, sel in selectors:
                try:
                    el = driver.find_element(by, sel)
                    if el:
                        return el
                except Exception as e:
                    last_err = e
            time.sleep(0.4)
        if last_err:
            raise last_err
        raise TimeoutError("No selector matched within timeout")

    try:
        file_input = first_present(file_selectors, timeout=40)
    except Exception:
        with open("deeptmhmm_debug_page.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        raise TimeoutError("Could not locate file input. Saved page to deeptmhmm_debug_page.html")

    file_input.send_keys(str(fasta_path.resolve()))
    print("    • File selected.")

    # Consent checkbox (if present)
    try:
        consent = first_present([
            (By.CSS_SELECTOR, "input[type='checkbox'][name*='consent']"),
            (By.CSS_SELECTOR, "input[type='checkbox'][id*='consent']"),
            (By.XPATH, "//input[@type='checkbox' and contains(@name,'consent')]"),
        ], timeout=5)
        if not consent.is_selected():
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", consent)
            time.sleep(0.3)
            consent.click()
            print("    • Consent checkbox ticked.")
    except Exception:
        pass

    # 2) Submit
    submit_selectors = [
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.CSS_SELECTOR, "input[type='submit']"),
        (By.XPATH, "//button[contains(.,'Run') or contains(.,'Predict') or contains(.,'Submit')]"),
        (By.XPATH, "//input[@type='submit' and (contains(@value,'Run') or contains(@value,'Predict') or contains(@value,'Submit'))]"),
    ]
    try:
        btn = first_present(submit_selectors, timeout=20)
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
        time.sleep(0.4)
        btn.click()
        print("    • Clicked submit button.")
    except Exception:
        try:
            form = driver.find_element(By.TAG_NAME, "form")
            driver.execute_script("arguments[0].submit();", form)
            print("    • Submitted form via JS fallback.")
        except Exception:
            with open("deeptmhmm_debug_page_after_upload.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            raise TimeoutError("Could not submit the job. Saved page to deeptmhmm_debug_page_after_upload.html")

    # 3) Wait for results UI
    WebDriverWait(driver, RESULT_WAIT_TIMEOUT).until(
        EC.any_of(
            EC.presence_of_element_located((By.XPATH, "//*[contains(.,'Predicted Topologies')]")),
            EC.presence_of_element_located((By.XPATH, "//*[contains(.,'Predictions')]")),
            EC.presence_of_element_located((By.XPATH, "//*[contains(.,'Results')]")),
        )
    )
    # Nudge lazy content
    for _ in range(6):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight/6)"); time.sleep(0.5)

    print("    • Results page detected.")

    # 4) Parse from page text (robust, no downloads needed)
    page_text = driver.find_element(By.TAG_NAME, "body").text

    three_line = extract_3line(page_text)
    gff3       = extract_gff3(page_text)

    chunk_dir = download_root / chunk_name
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if three_line:
        save_text(chunk_dir / f"{chunk_name}.3line", three_line)
        print(f"    • Saved {chunk_name}.3line")

    if gff3:
        save_text(chunk_dir / f"{chunk_name}.gff3", gff3)
        print(f"    • Saved {chunk_name}.gff3")

    if not three_line and not gff3:
        # Dump HTML for debugging if parsing failed
        with open(f"{chunk_name}_debug_results.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        raise RuntimeError(f"Could not parse results text for {chunk_name}; saved {chunk_name}_debug_results.html")

    print("    • Parsed results saved.\n")

def main():
    driver = build_driver(RESULT_DIR)
    try:
        chunks = sorted(CHUNK_DIR.glob("chunk_*.fa"))
        if not chunks:
            print("No chunks found. Make sure CHUNK_DIR has files like chunk_0001.fa")
            return

        for fa in chunks:
            submit_one_chunk(driver, fa, RESULT_DIR)
            time.sleep(2)  # polite delay

    finally:
        driver.quit()
        print("Done.")

if __name__ == "__main__":
    main()
