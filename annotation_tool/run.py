#------------------------------------------------------------
# START THE COLLABORATIVE TOOL ON LOCALHOST:9000
#------------------------------------------------------------
import argparse
import sys
from pathlib import Path

# Also allow: python annotation_tool/run.py from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation_tool.app import create_app
from annotation_tool.settings import read_settings


def main(argv=None):
    """Run one shared server, database and background inference worker."""
    import uvicorn
    parser = argparse.ArgumentParser(description="Shared fridge annotation workspace.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--port", type=int, help="Override the default port 9000.")
    parser.add_argument("--model-config", help="Shared inference YAML, including saved model paths.")
    args = parser.parse_args(argv)
    settings = read_settings(args.config)
    if args.model_config:
        settings["model_config"] = str(Path(args.model_config).resolve())
    app = create_app(settings)
    # One process owns the model worker; browser clients still edit concurrently.
    uvicorn.run(app, host=settings["host"], port=args.port or settings["port"], workers=1)


if __name__ == "__main__":
    main()
