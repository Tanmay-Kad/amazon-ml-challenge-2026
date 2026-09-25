"""
Normalization Module Verification & Test Suite
Amazon ML Challenge 2026 - Business Entity Resolution

Tests the normalization and tokenization functions from `src/normalization.py`
against real records sampled from `output/eda_samples/` and synthetic edge cases.
"""

import sys
import os
from pathlib import Path

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src to sys.path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalization import normalize_name, normalize_address, tokenize


class DualLogger:
    """Tee logger to stream test output both to console and a text file."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_log_path = output_dir / "normalization_test_output.txt"

    logger = DualLogger(test_log_path)
    sys.stdout = logger

    print("================================================================================")
    print("NORMALIZATION MODULE TEST SUITE")
    print("Amazon ML Challenge 2026: Business Entity Resolution")
    print("================================================================================")
    print(f"Log Output Path: {test_log_path}\n")

    # -------------------------------------------------------------------------
    # PART 1: Real Examples Pulled from 50k Reservoir Samples (output/eda_samples/)
    # -------------------------------------------------------------------------
    real_test_cases = [
        {
            "id": "REAL-01",
            "source": "train_source1",
            "category": "Pvt Ltd suffix + Indian street/road address",
            "name": "Sree Foundation Pvt Ltd",
            "address": "B-2-15, Hill View C.H.S., Pokhran Road No. 1, Devi Daya Nagar, Thane, Maharashtra",
        },
        {
            "id": "REAL-02",
            "source": "train_source1",
            "category": "Apostrophe in name + Drive abbreviation + US State",
            "name": "Orelee's Barbershop",
            "address": "1795 Westchester Drive, High Point, NC",
        },
        {
            "id": "REAL-03",
            "source": "train_source1",
            "category": "Punctuation in name (+ symbol) + Inc suffix + Avenue abbreviation",
            "name": "B+ Retail Inc",
            "address": "1712 Montebello Avenue, Phoenix, AZ",
        },
        {
            "id": "REAL-04",
            "source": "train_source1",
            "category": "Unit/Apartment address component with mixed casing",
            "name": "Christ Chapel",
            "address": "2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD",
        },
        {
            "id": "REAL-05",
            "source": "train_source2",
            "category": "EMPTY address (handling 2.6-3.4% missing rate from EDA)",
            "name": "Smt Leather Public School Ltd",
            "address": "",
        },
        {
            "id": "REAL-06",
            "source": "train_source2",
            "category": "Leading punctuation dashes + Peak Inc suffix + Street abbreviation",
            "name": "-- Holloway Peak Inc Seafood",
            "address": "105 ELM ST, MORGANTON, NC",
        },
        {
            "id": "REAL-07",
            "source": "train_source2",
            "category": "Hindi Devanagari script (preserving script without crash)",
            "name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        },
        {
            "id": "REAL-08",
            "source": "train_source2",
            "category": "Bengali script name and address (preserving non-Latin script)",
            "name": "লোটাস আইটি প্রাইভেট লিমিটেড",
            "address": "2 NO DHAPA RD, KOLKATA, HOWRAH, পশ্চিমবঙ্গ",
        },
        {
            "id": "REAL-09",
            "source": "train_source2",
            "category": "All-caps PRIVATE LTD + Indian industrial plaza address",
            "name": "SURAT SOLUTIONS PRIVATE LTD",
            "address": "PLOT 106 SHOP-235, HIGHFIELD ASCOT OPP-PALM AVENUE, VESU, SURAT, Gujarat",
        },
        {
            "id": "REAL-10",
            "source": "train_source2",
            "category": "All-caps LLC with comma punctuation + Street abbreviation",
            "name": "LY CARDIOLOGY, LLC",
            "address": "194 Jones St, SUN PRAIRIE, WI",
        },
        {
            "id": "REAL-11",
            "source": "train_source3",
            "category": "EMPTY address with trailing plus in name",
            "name": "Gyanmurti  +",
            "address": "",
        },
        {
            "id": "REAL-12",
            "source": "train_source3",
            "category": "Acute accent in US company name (Phármaceuticals) + leading ## in address",
            "name": "0ppenheimer Phármaceuticals Associates",
            "address": "##29656 390th Ave, Grygla, Minnesota",
        },
        {
            "id": "REAL-13",
            "source": "train_source3",
            "category": "LLC prefix placement + Street abbreviation",
            "name": "Vanderpool LLC Center",
            "address": "52 Broad St, Story City, Iowa",
        },
        {
            "id": "REAL-14",
            "source": "test_source1",
            "category": "French SARL with & ampersand and grave accent (Frères)",
            "name": "Cobayes & Frères SARL",
            "address": "47 Cité Dutrey, Bordeaux, Nouvelle-Aquitaine",
        },
        {
            "id": "REAL-15",
            "source": "test_source1",
            "category": "French legal entity (SARL) + European street format (Rue / bis)",
            "name": "ZNB Club SARL",
            "address": "Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac",
        },
        {
            "id": "REAL-16",
            "source": "test_source2",
            "category": "French SCI legal prefix + grave accent in name (Àmicale)",
            "name": "SCI Ptit Àmicale",
            "address": "18 RUE JEN ZAY, Dunkerque, Nord",
        },
    ]

    # -------------------------------------------------------------------------
    # PART 2: Synthetic Edge-Case Tests (Robustness Verification)
    # -------------------------------------------------------------------------
    edge_test_cases = [
        {
            "id": "EDGE-01",
            "category": "Empty string input for both name and address",
            "name": "",
            "address": "",
        },
        {
            "id": "EDGE-02",
            "category": "Whitespace-only input with mixed spaces, tabs, and newlines",
            "name": "   \t  \n  ",
            "address": "  \t  ",
        },
        {
            "id": "EDGE-03",
            "category": "Purely non-Latin Devanagari script with NO legal suffixes or Latin chars",
            "name": "आदित्य प्रॉपर्टीज",
            "address": "गुलमोहर कॉलोनी, भोपाल, मध्य प्रदेश",
        },
        {
            "id": "EDGE-04",
            "category": "None / non-string type input gracefully handled",
            "name": None,
            "address": None,
        },
        {
            "id": "EDGE-05",
            "category": "Multiple punctuation dots and abbreviation chains (Pvt. Ltd. & Co.)",
            "name": "A.B.C. Pvt. Ltd. & Co.",
            "address": "N. Main St., Suite #400, N.W.",
        },
    ]

    total_tests_run = 0
    exceptions_encountered = 0

    print("--- 1. Testing Hand-Picked Real Examples from Dataset Samples ---")
    for tc in real_test_cases:
        total_tests_run += 1
        try:
            norm_name = normalize_name(tc["name"])
            norm_addr = normalize_address(tc["address"])
            name_tokens = tokenize(norm_name)
            addr_tokens = tokenize(norm_addr)

            print(f"[{tc['id']}] {tc['category']} (Source: {tc['source']})")
            print(f"  Raw Name:        {tc['name']!r}")
            print(f"  Normalized Name: {norm_name!r}")
            print(f"  Name Tokens:     {name_tokens}")
            print(f"  Raw Address:     {tc['address']!r}")
            print(f"  Normalized Addr: {norm_addr!r}")
            print(f"  Address Tokens:  {addr_tokens}")
            print()
        except Exception as e:
            exceptions_encountered += 1
            print(f"FAILED on {tc['id']}: {e}\n")

    print("\n--- 2. Testing Synthetic Edge Cases & Robustness ---")
    for tc in edge_test_cases:
        total_tests_run += 1
        try:
            norm_name = normalize_name(tc["name"])
            norm_addr = normalize_address(tc["address"])
            name_tokens = tokenize(norm_name)
            addr_tokens = tokenize(norm_addr)

            print(f"[{tc['id']}] {tc['category']}")
            print(f"  Raw Name:        {tc['name']!r}")
            print(f"  Normalized Name: {norm_name!r}")
            print(f"  Name Tokens:     {name_tokens}")
            print(f"  Raw Address:     {tc['address']!r}")
            print(f"  Normalized Addr: {norm_addr!r}")
            print(f"  Address Tokens:  {addr_tokens}")
            print()
        except Exception as e:
            exceptions_encountered += 1
            print(f"FAILED on {tc['id']}: {e}\n")

    # Final summary check
    print("================================================================================")
    print("TEST SUITE SUMMARY")
    print("================================================================================")
    print(f"Total Test Cases Executed: {total_tests_run} (Real: {len(real_test_cases)}, Edge: {len(edge_test_cases)})")
    print(f"Exceptions / Crashes:      {exceptions_encountered}")
    print(f"All Tests Passed:          {exceptions_encountered == 0}")
    print("================================================================================")

    sys.stdout = logger.terminal
    logger.close()

    if exceptions_encountered > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
