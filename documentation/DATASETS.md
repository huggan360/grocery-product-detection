# Installed datasets and citations

The data lives under `data/`, which is excluded from Git. No models have been trained on it. The BibTeX entries are in [references.bib](references.bib).

[dataset_sources.json](dataset_sources.json) records the downloaded revision, actual counts, and SHA-256 hashes of the annotation files and API responses.

## Swedish Grocery Store Dataset

- Source: [official repository](https://github.com/marcusklasson/GroceryStoreDataset).
- Paper: [A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels](https://arxiv.org/abs/1901.00711), Klasson, Zhang and Kjellström, WACV 2019.
- Downloaded revision: `fc80ba90f803d79d0383df52c5a4ac5de99ff6fc` on 2026-09-23.
- Raw files: `data/raw/grocery-store-dataset/`.
- Prepared classifier folders: `data/grocery-store/classification/`.
- Reproduce preparation: `python import_grocery_store.py` with a new/empty output folder.
- Repository license: MIT; original `LICENSE` is retained in the raw download.
- Citation keys: `klasson2019hierarchical` and optionally `klasson_grocerystore_dataset_snapshot`.

This research dataset contains grocery photos collected in Stockholm stores. It includes natural images and separate iconic product images/descriptions. Only the natural images listed in the official split files are used in the prepared classifier folders.

The downloaded files contain **5,421 labelled natural images and 43 coarse categories**:

| Split | Images | Categories represented |
| --- | ---: | ---: |
| Training | 2,640 | 43 |
| Validation | 296 | 37 |
| Test | 2,485 | 43 |

These measured counts differ from the repository README's older summary. The import report is `data/grocery-store/classification/SOURCE.json`; it includes per-category counts. Every referenced image was decoded and checked for identical pixels across splits during import. The official splits were kept unchanged.

The validation split lacks garlic, nectarine, papaya, plum, sour-milk and soy-milk. Classifier training warns about these gaps. Validation cannot estimate recall for those categories; the test set must remain untouched during tuning. Near-duplicate scenes are not detected by the exact-pixel audit.

This is **classification data without object boxes**. It is useful for ViT initialization but does not directly train the YOLO detector. Some images have surrounding products, so its image-level labels are not a substitute for bounding-box annotations. Its categories do not cover all groceries; for example, it does not supply the full requested butter/bread/ketchup/salsa category set.

Use `configs/swedish_grocery.yaml` when you are ready to train the classifier.

## Open Food Facts: products sold in Sweden

- Source: [Open Food Facts API](https://openfoodfacts.github.io/openfoodfacts-server/api/).
- Filter: `countries_tags=en:sweden`; two pages of up to 100 product records, ordered by `unique_scans_n`.
- Local files: `data/raw/open-food-facts-sweden/`.
- Raw API pages: `page_1.json` and `page_2.json`.
- Downloaded images: `images/`.
- Review table: `review.csv`, with barcode, name, original category tags, suggested category, product URL and image URL.
- Actual counts and any download failures: `SOURCE.json`.
- Installed sample: **200 product records and 200 images**, with no failed image downloads.
- Citation key: `openfoodfacts_sweden_2026`.

“Swedish-market” means listed as sold in Sweden; it does not imply Swedish manufacture. This is a small convenience sample, not a balanced benchmark. Community metadata and category suggestions need review. It is kept separate from the prepared training folders so unchecked labels do not silently enter training.

To use this collection, review each image and assign one broad category, then place approved images in the classification folder structure described in the README. Keep all images of the same barcode in one split. If mixing this source with other datasets, also check for the same product photos across sources. These images do not have shelf bounding boxes.

The [source documentation](https://openfoodfacts.github.io/openfoodfacts-server/api/) specifies ODbL for the database, Database Contents License for individual database contents, and [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) for images. Attribute Open Food Facts contributors and preserve product/image URLs. The local JPEG copies are RGB conversions of the source images; original API responses are retained.

## Cite in LaTeX

For traditional BibTeX:

```latex
We use the Grocery Store Dataset~\cite{klasson2019hierarchical}.
Additional Swedish-market product images come from Open Food
Facts~\cite{openfoodfacts_sweden_2026}.

\bibliographystyle{IEEEtran}
\bibliography{references}
```

Only cite the Open Food Facts subset as experimental data if you actually include it in your experiments. Downloading a dataset alone does not mean it was used to train or evaluate a model.
