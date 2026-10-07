#------------------------------------------------------------
# DOWNLOAD A SMALL SWEDISH-MARKET PRODUCT IMAGE COLLECTION
#------------------------------------------------------------
import argparse
import csv
import io
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

CATEGORY_TAGS = {
    "en:ketchups": "ketchup", "en:salsa-sauces": "salsa", "en:butters": "butter",
    "en:milks": "milk", "en:breads": "bread", "en:yogurts": "yoghurt",
    "en:cheeses": "cheese", "en:creams": "cream", "en:eggs": "egg",
    "en:fruit-juices": "juice", "en:apples": "apple", "en:bananas": "banana",
}


def download_products(output, pages=2):
    """Save Swedish-market photos and their original labels for manual review."""
    if not 1 <= pages <= 3:
        raise ValueError("Use 1 to 3 pages. Use official bulk exports for larger collections.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "GroceryProductDetection-CourseProject/0.1 (educational dataset download)"
    fields = "code,product_name,brands,countries_tags,categories_tags,image_front_url,url"
    products = {}
    for page in range(1, pages + 1):
        cached = output / f"page_{page}.json"
        if cached.exists():
            payload = json.loads(cached.read_text())
        else:
            response = session.get("https://world.openfoodfacts.org/api/v2/search", params={
                "countries_tags": "en:sweden", "page_size": 100, "page": page,
                "fields": fields, "sort_by": "unique_scans_n",
            }, timeout=60)
            response.raise_for_status()
            payload = response.json()
            cached.write_text(json.dumps(payload, indent=2) + "\n")
            # Stay below the documented limit of ten search requests per minute.
            time.sleep(6.1)
        for product in payload.get("products", []):
            if "en:sweden" in product.get("countries_tags", []):
                products[product["code"]] = product
    rows, failures = [], []
    for code, product in products.items():
        url = product.get("image_front_url", "")
        parsed = urlparse(url)
        if not code.isdigit() or parsed.scheme != "https" or parsed.hostname != "images.openfoodfacts.org":
            continue
        target = images / f"{code}.jpg"
        try:
            if not target.exists():
                response = session.get(url, timeout=30)
                response.raise_for_status()
                with Image.open(io.BytesIO(response.content)) as image:
                    image.convert("RGB").save(target, quality=95)
                time.sleep(0.2)
            with Image.open(target) as image:
                image.verify()
        except (requests.RequestException, OSError) as error:
            failures.append({"code": code, "error": str(error)})
            continue
        tags = product.get("categories_tags", [])
        suggestions = list(dict.fromkeys(category for tag, category in CATEGORY_TAGS.items() if tag in tags))
        rows.append({"image": str(target.relative_to(output)), "barcode": code,
                     "product_name": product.get("product_name", ""), "brands": product.get("brands", ""),
                     "suggested_category": suggestions[0] if len(suggestions) == 1 else "",
                     "review_status": "needs_review", "source_url": f"https://world.openfoodfacts.org/product/{code}",
                     "image_url": url, "original_category_tags": "|".join(tags)})
        if len(rows) % 20 == 0:
            print(f"Downloaded/checked {len(rows)} images", flush=True)
    if not rows:
        raise RuntimeError("No product images downloaded. Check API responses in the output folder.")
    with (output / "review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source": "https://world.openfoodfacts.org", "filter": "countries_tags=en:sweden",
        "meaning": "Products listed as sold in Sweden; not necessarily produced in Sweden.",
        "images": len(rows), "products_returned": len(products), "failures": failures,
        "database_license": "ODbL 1.0", "contents_license": "Database Contents License 1.0",
        "image_license": "CC BY-SA 3.0", "attribution": "Open Food Facts contributors",
        "license_source": "https://openfoodfacts.github.io/openfoodfacts-server/api/",
        "usage": "Review category suggestions before using for training. No bounding boxes are supplied.",
    }
    (output / "SOURCE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


#------------------------------------------------------------
# RUN THE DOWNLOADER ONLY WHEN REQUESTED
#------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download up to 300 Swedish-market Open Food Facts products.")
    parser.add_argument("--output", default="data/raw/open-food-facts-sweden")
    parser.add_argument("--pages", type=int, default=2)
    args = parser.parse_args()
    download_products(args.output, args.pages)
