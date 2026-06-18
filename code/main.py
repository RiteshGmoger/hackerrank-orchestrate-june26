import argparse
import csv
import random
from pathlib import Path
from agent import run_agent
from schema import AgentOutput
from config import INPUT_CSV, OUTPUT_CSV, SAMPLE_CSV

random.seed(42)


def run(input_path: Path, output_path: Path) -> None:
    """
        Main pipeline loop.Flushes after each row
        Why flush: if process crashes at ticket 25 of 29, first 24 rows are saved
    """
    fields = ["ticket_id", "status", "product_area", "response", "justification", "request_type"]
    results = []

    with open(input_path, newline = "", encoding = "utf-8") as inf, \
         open(output_path, "w", newline = "", encoding = "utf-8") as outf:

        reader = csv.DictReader(inf)
        writer = csv.DictWriter(outf, fieldnames=fields)
        writer.writeheader()

        for i, row in enumerate(reader, 1):
            print(f"[{i}] {row.get('ticket_id', 'unknown')}...", end = " ", flush = True)
            try:
                result = run_agent(row)
            except Exception as e:
                # main: never let one ticket crash the whole run.
                # Wrap in escalation so output.csv always has a complete row
                result = AgentOutput(
                    ticket_id = row.get("ticket_id", f"row_{i}"),
                    status = "escalated",
                    product_area = "unknown",
                    response = "",
                    justification = f"Unhandled agent exception: {str(e)}",
                    request_type = "product_issue"
                )
            results.append(result)
            writer.writerow(result.model_dump())
            outf.flush()
            print(f"{result.status} | {result.product_area}")

    if results:
        replied = sum(1 for r in results if r.status == "replied")
        escalated = len(results) - replied
        print(f"\n{'='*40}")
        print(f"COMPLETE: {len(results)} tickets processed")
        print(f"  replied:   {replied} ({100*replied//len(results)}%)")
        print(f"  escalated: {escalated} ({100*escalated//len(results)}%)")
        print(f"{'='*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action = "store_true")
    parser.add_argument("--input", default = None)
    parser.add_argument("--output", default = None)
    args = parser.parse_args()

    if args.sample:
        input_path = Path(SAMPLE_CSV)
    elif args.input:
        input_path = Path(args.input)
    else:
        input_path = Path(INPUT_CSV)

    output_path = Path(args.output) if args.output else Path(OUTPUT_CSV)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    run(input_path, output_path)
    print(f"Done. Output -> {output_path}")
