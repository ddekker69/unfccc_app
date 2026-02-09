"""
Example: Using the New Error Handling Features in extract_headings_and_sections.py

This script demonstrates how to use the new error handling capabilities:
- run_one_file_safe() - Returns ExtractionResult instead of raising exceptions
- generate_error_report() - Creates human-readable error reports
- save_error_summary() - Aggregates errors across multiple documents

Run this script to see error handling in action!
"""

from pathlib import Path
from extract_headings_and_sections import (
    run_one_file_safe,
    generate_error_report,
    save_error_summary
)

def main():
    print("=" * 80)
    print("DEMONSTRATION: New Error Handling Features")
    print("=" * 80)

    # Example 1: Process a valid document
    print("\n📄 Example 1: Processing a valid document")
    print("-" * 80)

    result = run_one_file_safe('enb1201e')

    if result.success:
        print(f"✓ SUCCESS: Processed {result.stats['doc_code']}")
        print(f"  Pages: {result.stats['num_pages']}")
        print(f"  Headings: {result.stats['num_headings']}")
        print(f"  Sections: {result.stats['num_sections']}")
        print(f"  Time: {result.stats['processing_time']:.2f}s")
    else:
        print(f"✗ FAILED: Could not process document")

    if result.warnings:
        print(f"\n⚠️  Warnings: {len(result.warnings)}")
        for warn in result.warnings:
            print(f"  - {warn.message}")

    # Example 2: Handle a non-existent file
    print("\n\n📄 Example 2: Handling a non-existent file")
    print("-" * 80)

    result = run_one_file_safe('nonexistent_file')

    if not result.success:
        print("✗ Expected failure: File doesn't exist")
        print(f"  Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"  - Type: {err.error_type}")
            print(f"    Message: {err.message}")
            if err.recovery_action:
                print(f"    Recovery: {err.recovery_action}")

    # Example 3: Generate detailed error report
    print("\n\n📄 Example 3: Generating detailed error report")
    print("-" * 80)

    report = generate_error_report(result)
    print(report)

    # Example 4: Process multiple documents and generate summary
    print("\n\n📄 Example 4: Processing multiple documents with summary")
    print("-" * 80)

    doc_codes = ['enb1201e', 'enb1202e', 'enb1203e', 'nonexistent']
    results = []

    for doc_code in doc_codes:
        print(f"Processing {doc_code}...", end=" ")
        result = run_one_file_safe(doc_code)
        results.append(result)

        if result.success:
            print(f"✓ ({result.stats['num_headings']} headings)")
        else:
            print(f"✗ ({len(result.errors)} errors)")

    # Save aggregate summary
    output_dir = Path('unfccc_processed_data_v2')
    save_error_summary(results, output_dir)

    print(f"\n📊 Summary saved to {output_dir}/error_summary.txt")

    # Show summary
    summary_file = output_dir / 'error_summary.txt'
    if summary_file.exists():
        print("\n" + "=" * 80)
        print("AGGREGATE SUMMARY")
        print("=" * 80)
        with open(summary_file) as f:
            print(f.read())

    print("\n" + "=" * 80)
    print("✅ Demonstration complete!")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("1. run_one_file_safe() never raises exceptions - always returns ExtractionResult")
    print("2. Check result.success to know if processing succeeded")
    print("3. result.errors contains all errors with detailed context")
    print("4. result.warnings contains non-critical issues")
    print("5. result.stats contains processing metrics")
    print("6. Use generate_error_report() for human-readable output")
    print("7. Use save_error_summary() to aggregate errors across documents")

    print("\n💡 Benefits:")
    print("- No more unexpected crashes")
    print("- Partial results when possible")
    print("- Detailed error tracking")
    print("- Recovery suggestions")
    print("- Batch processing friendly")

if __name__ == "__main__":
    main()
