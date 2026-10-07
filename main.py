#------------------------------------------------------------
# RUN PREDICTIONS: ACQUISITION OR PRODUCTION
#------------------------------------------------------------
import argparse
from pathlib import Path


def main():
    """Read one capture request and write one JSON result for the orchestrator."""
    parser = argparse.ArgumentParser(description="Run YOLO masks and optional ViT classification.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data-aquisition", "--data-acquisition", dest="acquisition", action="store_true")
    mode.add_argument("--production", action="store_true")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--record-id", required=True, help="SQLite capture ID matching the request.")
    parser.add_argument("--output", type=Path, required=True, help="Result JSON in a new run directory.")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/inference.yaml")
    parser.add_argument("--models-dir", type=Path, help="Directory containing yolo/ and vit/.")
    parser.add_argument("--managed-retention", action="store_true", help="The orchestrator owns cleanup.")
    args = parser.parse_args()
    from production.run import run_request
    print(run_request(args.request, args.record_id, args.output, args.config,
                      "acquisition" if args.acquisition else "production",
                      models_dir=args.models_dir, managed_retention=args.managed_retention))


if __name__ == "__main__":
    main()
