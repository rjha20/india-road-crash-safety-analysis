# Source data handling

Raw source data is kept local under `data/raw/` and excluded from Git. The project uses source manifests with retrieval URL, version, license/terms note, retrieval date, checksum, and local filename. Do not publish raw event rows, newspaper article bodies, personal/victim details, or dataset archives with the project.

## Mendeley v7 structured files

- Dataset: [Media Reported Road Traffic Crash Data](https://data.mendeley.com/datasets/bc5sv6wnd9/7), DOI `10.17632/bc5sv6wnd9.7`, version 7 (published 27 January 2025).
- Landing page declares CC BY 4.0 for the dataset. The source article content may have separate copyright; do not redistribute raw article text.
- Locally obtained structured files: `v7/news_crashes.xlsx` and `v7/data_dictionary.xlsx`.
- Locally obtained 2022 text file: `v7/raw_articles/2022_raw_data.csv`. Article text must not be shown in reports/dashboard or checked into Git. 2023 raw text download returned a server-error JSON during this build; the failed response is not a dataset and is not used.
- Keep filenames and source IDs in the manifest; never treat incomplete raw text coverage as full 2022–2023 NLP coverage.

## MoRTH report

The official [Road Accidents in India 2023 report](https://www.morth.gov.in/sites/default/files/Road-Accident-in-India-2023-Publications.pdf) supplies aggregate context. A derived table must retain report title, page/table, year, state, numerator definition, and transformations. Do not re-distribute the full PDF or inferred/unverified values without checking the ministry's terms.

## RDD2022 India image benchmark

Source: [RDD2022 Figshare record](https://doi.org/10.6084/m9.figshare.21431547.v1), DOI `10.6084/m9.figshare.21431547.v1`. Figshare reports CC BY 4.0. Challenge material may include CC BY-SA 4.0 terms; preserve attribution and review conditions for the precise archive used. Do not claim this dataset is geolinked to media-reported crashes. Use the image IDs and original split/country metadata in evaluation.

This build selectively extracted only the India/train JPEG and VOC XML members using verified byte-range reads; the full 13.26 GB multi-country archive was not saved. The local subset contains 7,706 images and 7,706 XML files. See `rdd2022/source_manifest.json` and `data/processed/vision/remote_archive_profile.json` for archive metadata and extraction scope. The model further excludes images with out-of-scope/malformed damage labels from its four-class train/validation split.
