# Live Surface: av-map-data

This file documents the currently active production surface area for the `av-map-data` repo.

## Production Entrypoints

- Canonical data files:
  - `events.csv`
  - `geometries/*.geojson`
- CI pipeline: `.github/workflows/deploy-data.yml`

## Active Runtime Flow

1. `validate` job runs `pytest tests/ -v`
2. `update-cache` job runs on non-PR events after validation
3. CSV import: `.dev/import-csv.py`
4. Geometry upload: `.dev/upload-geometries.js`
5. Geometry metadata sync: `.dev/sync-geometries-table.js`
6. Cache rebuild and upload: `.dev/rebuild-cache.js`

## Critical Dependencies

- GitHub Actions for validation and publish automation
- Supabase database tables (`av_events*`, geometry metadata tables)
- Supabase storage buckets (cache + geometry buckets)
- Python 3.11 + Node 20 in CI

## Active Environment Variables

- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `STAGING` (controls staging vs production target)
- `GITHUB_ACTIONS` (CI behavior toggle)

## Source of Truth for Data Contract

- Schema contract: `.dev/schema.json`
- Validation tests: `tests/test_validation.py`
- Contributor guidance: `CONTRIBUTING.md`

These three must remain aligned for event types, required fields, and geometry naming rules.
