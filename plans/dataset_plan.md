I want to build a massive dataset for research purposes. I wont be able to release this model for commercial usage, but will be able to conduct lots of tests until i scale up my open source database. for now this is likely the fastest way to go.

to do this i would like music in a variety of genres

and we need 100% certainty that all the music does not have vocals.

best way to do this is just through youtube scraping

ill gather the txt files per genre first

## Jamendo Harvest Plan

- Use the Jamendo API instead of blind scraping so each track arrives with license + metadata.
- Script entrypoint (max mode): `python scripts/jamendo_scraper.py --client-id $JAMENDO_CLIENT_ID --output-dir data/jamendo_full`.
- If you want to stop early per genre, add `--per-genre-limit N`; otherwise it pages until Jamendo runs out of tracks.
- By default the scraper loops over a 94-genre list derived from the MTG-Jamendo dataset taxonomy (covers the canonical Jamendo tag space); pass `--genres ...` or `--genres-file ...` to scope it down or expand it.
- Add `--instrumental-only` if you want Jamendo’s own instrumental flag applied; otherwise everything is captured and you filter vocals locally.
- Each run writes `<genre>.jsonl` metadata plus `<genre>_urls.txt` download manifests that feed `scripts/download_music.py`.
- Re-run as needed to grow beyond the 10k hour target; the summary JSON shows track counts per genre/tag.