#!/usr/bin/env python3
"""
Pytest tests for AV Map Data validation.
Converts the validation script into pytest-compatible tests.
"""

import csv
import json
import pytest
import warnings
from pathlib import Path
import re
from urllib.parse import urlparse
from schema_loader import get_schema, get_event_types, get_column_names, get_enum_values, get_required_fields


@pytest.fixture
def repo_root():
    """Get the repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def csv_file(repo_root):
    """Get the events CSV file path."""
    return repo_root / 'events.csv'


@pytest.fixture
def geometries_dir(repo_root):
    """Get the geometries directory path."""
    return repo_root / 'geometries'


def test_csv_file_exists(csv_file):
    """Test that the events.csv file exists."""
    assert csv_file.exists(), "CSV file not found: events.csv"


def test_csv_headers(csv_file):
    """Test that CSV has correct headers."""
    expected_headers = [
        'date', 'event_type', 'company', 'city', 'geometry_file',
        'vehicles', 'platform', 'fares', 'direct_booking', 'service_model', 'supervision',
        'access', 'fleet_partner', 'expected_launch', 'company_link', 'booking_platform_link', 'source_url', 'notes'
    ]

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_headers, \
            f"CSV headers incorrect. Expected: {expected_headers}, Got: {reader.fieldnames}"


def test_csv_required_fields(csv_file):
    """Test that required fields are present in all rows."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            if not row.get('date'):
                errors.append(f"Row {row_num}: Missing date")

            if not row.get('event_type'):
                errors.append(f"Row {row_num}: Missing event_type")

    assert len(errors) == 0, "\n".join(errors)


def test_date_format(csv_file):
    """Test that dates are in YYYY-MM-DD format."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            date_val = row.get('date', '')
            if date_val and not re.match(r'^\d{4}-\d{2}-\d{2}$', date_val):
                errors.append(f"Row {row_num}: Invalid date format. Expected YYYY-MM-DD, got: {date_val}")

    assert len(errors) == 0, "\n".join(errors)


def test_event_types(csv_file):
    """Test that event types are valid."""
    valid_event_types = get_event_types()
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            event_type = row.get('event_type', '')
            if event_type and event_type not in valid_event_types:
                errors.append(f"Row {row_num}: Invalid event_type '{event_type}'. Valid types: {valid_event_types}")

    assert len(errors) == 0, "\n".join(errors)


def test_service_created_events(csv_file):
    """Test that service_created events have all required fields."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            if row.get('event_type') == 'service_created':
                # Note: 'platform' is optional - some services don't have booking platforms initially
                required_fields = ['company', 'city', 'vehicles', 'fares',
                                 'direct_booking', 'service_model', 'supervision', 'access']
                for field in required_fields:
                    if not row.get(field, '').strip():
                        errors.append(f"Row {row_num}: service_created event missing required field: {field}")

    assert len(errors) == 0, "\n".join(errors)


def test_service_ended_events(csv_file):
    """Test that service_ended events are correctly formatted."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            if row.get('event_type') == 'service_ended':
                if not row.get('company', '').strip():
                    errors.append(f"Row {row_num}: service_ended event missing required field: company")
                if not row.get('city', '').strip():
                    errors.append(f"Row {row_num}: service_ended event missing required field: city")

                service_fields = ['vehicles', 'platform', 'fares', 'direct_booking',
                                'supervision', 'access', 'fleet_partner']
                filled_fields = [field for field in service_fields if row.get(field, '').strip()]
                if len(filled_fields) > 0:
                    errors.append(f"Row {row_num}: service_ended event should not have service attribute fields filled: {filled_fields}")

    assert len(errors) == 0, "\n".join(errors)


def test_service_testing_events(csv_file):
    """Test that service_testing events have required fields."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            if row.get('event_type') == 'service_testing':
                # Only require company, city, source_url
                if not row.get('company', '').strip():
                    errors.append(f"Row {row_num}: service_testing event missing required field: company")
                if not row.get('city', '').strip():
                    errors.append(f"Row {row_num}: service_testing event missing required field: city")
                if not row.get('source_url', '').strip():
                    errors.append(f"Row {row_num}: service_testing event missing required field: source_url")

    assert len(errors) == 0, "\n".join(errors)


def test_service_announced_events(csv_file):
    """Test that service_announced events have required fields."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            if row.get('event_type') == 'service_announced':
                # Only require company, city, source_url
                if not row.get('company', '').strip():
                    errors.append(f"Row {row_num}: service_announced event missing required field: company")
                if not row.get('city', '').strip():
                    errors.append(f"Row {row_num}: service_announced event missing required field: city")
                if not row.get('source_url', '').strip():
                    errors.append(f"Row {row_num}: service_announced event missing required field: source_url")

    assert len(errors) == 0, "\n".join(errors)


def test_update_events(csv_file):
    """Test that update events are correctly formatted."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            event_type = row.get('event_type', '')
            if event_type.endswith('_updated') or event_type.endswith('_changed'):
                service_fields = ['vehicles', 'platform', 'fares', 'direct_booking',
                                'supervision', 'access', 'fleet_partner']
                filled_fields = [field for field in service_fields if row.get(field, '').strip()]

                if event_type == 'geometry_updated':
                    if len(filled_fields) > 0:
                        errors.append(f"Row {row_num}: geometry_updated event should not have service attribute fields filled")
                    if not row.get('geometry_file', '').strip():
                        errors.append(f"Row {row_num}: geometry_updated event must have geometry_file")
                else:
                    if len(filled_fields) == 0:
                        errors.append(f"Row {row_num}: {event_type} event should have at least one service attribute field filled")

                # Company field is REQUIRED for update events to identify which service in multi-company cities
                if not row.get('company', '').strip():
                    errors.append(f"Row {row_num}: Update event must have company field to identify the service")

    assert len(errors) == 0, "\n".join(errors)


def test_geometry_file_naming(csv_file):
    """Test that geometry files follow naming conventions or are valid inline coordinates."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            geometry_file = row.get('geometry_file', '')
            if geometry_file:
                # Check if it's inline coordinates (lng,lat format)
                if re.match(r'^-?\d+\.?\d*,-?\d+\.?\d*$', geometry_file):
                    # Valid inline coordinates - skip file naming validation
                    continue

                if not geometry_file.endswith('.geojson'):
                    errors.append(f"Row {row_num}: Geometry file should end with .geojson or be inline coordinates (lng,lat): {geometry_file}")

                # Pattern: company-city-month-day-year-boundary.geojson
                expected_pattern = r'^[a-z0-9]+-[a-z0-9-]+-[a-z]+-\d{1,2}-\d{4}-boundary\.geojson$'
                if not re.match(expected_pattern, geometry_file):
                    errors.append(f"Row {row_num}: Geometry file doesn't follow naming convention: {geometry_file}")

    assert len(errors) == 0, "\n".join(errors)


def test_service_attribute_values(csv_file):
    """Test that service attributes have valid values."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            # Validate enum fields using schema
            enum_fields = ['fares', 'direct_booking', 'service_model', 'supervision', 'access']
            for field in enum_fields:
                value = row.get(field, '').strip()
                if value:
                    valid_values = get_enum_values(field)
                    if value not in valid_values:
                        errors.append(f"Row {row_num}: {field} must be one of {valid_values}, got: {value}")

    assert len(errors) == 0, "\n".join(errors)


def test_url_format(csv_file):
    """Test that URLs are properly formatted."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            source_url = row.get('source_url', '')
            if source_url:
                try:
                    result = urlparse(source_url)
                    if not all([result.scheme, result.netloc]):
                        errors.append(f"Row {row_num}: Invalid URL format: {source_url}")
                except Exception:
                    errors.append(f"Row {row_num}: Invalid URL format: {source_url}")

    assert len(errors) == 0, "\n".join(errors)


def test_geometries_directory_exists(geometries_dir):
    """Test that the geometries directory exists."""
    assert geometries_dir.exists(), "Geometries directory not found"


def test_geojson_files_valid(geometries_dir):
    """Test that all GeoJSON files are valid."""
    errors = []
    geojson_files = list(geometries_dir.glob("*.geojson"))

    for geojson_file in geojson_files:
        try:
            with open(geojson_file, 'r') as f:
                geojson_data = json.load(f)

            geojson_type = geojson_data.get('type')

            # Accept both FeatureCollection and single Feature
            if geojson_type == 'FeatureCollection':
                if 'features' not in geojson_data:
                    errors.append(f"{geojson_file.name}: Missing 'features' array")
                    continue

                if not isinstance(geojson_data['features'], list):
                    errors.append(f"{geojson_file.name}: 'features' must be an array")
                    continue

                features = geojson_data['features']
            elif geojson_type == 'Feature':
                # Single feature - wrap in array for validation
                features = [geojson_data]
            else:
                errors.append(f"{geojson_file.name}: Must be a FeatureCollection or Feature, got: {geojson_type}")
                continue

            for i, feature in enumerate(features):
                if feature.get('type') != 'Feature':
                    errors.append(f"{geojson_file.name}: Feature {i} must have type 'Feature'")

                if 'geometry' not in feature:
                    errors.append(f"{geojson_file.name}: Feature {i} missing geometry")

                if 'properties' not in feature:
                    errors.append(f"{geojson_file.name}: Feature {i} missing properties")

                geometry = feature.get('geometry', {})
                geom_type = geometry.get('type')
                if geom_type not in ['Polygon', 'MultiPolygon', 'Point', 'LineString']:
                    errors.append(f"{geojson_file.name}: Feature {i} has invalid geometry type: {geom_type}")

        except json.JSONDecodeError as e:
            errors.append(f"{geojson_file.name}: Invalid JSON - {e}")
        except Exception as e:
            errors.append(f"{geojson_file.name}: Error reading file - {e}")

    assert len(errors) == 0, "\n".join(errors)


def test_geometry_file_references(csv_file, geometries_dir):
    """Test that all referenced geometry files exist."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            geometry_file = row.get('geometry_file', '').strip()
            if geometry_file:
                # Skip validation if it's inline coordinates (lng,lat format)
                if re.match(r'^-?\d+\.?\d*,-?\d+\.?\d*$', geometry_file):
                    continue

                geometry_path = geometries_dir / geometry_file
                if not geometry_path.exists():
                    errors.append(f"Row {row_num}: Referenced geometry file does not exist: {geometry_file}")

    assert len(errors) == 0, "\n".join(errors)


def test_data_consistency(csv_file):
    """Test data consistency for service_created events."""
    errors = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            event_type = row.get('event_type', '')
            if event_type == 'service_created':
                if not row.get('company'):
                    errors.append(f"Row {row_num}: service_created event missing company")
                if not row.get('city'):
                    errors.append(f"Row {row_num}: service_created event missing city")

    assert len(errors) == 0, "\n".join(errors)


def test_recommended_fields(csv_file):
    """Warn about missing recommended fields (non-failing)."""
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, start=2):
            # Warn if source_url is missing (recommended but not required)
            if not row.get('source_url', '').strip():
                warnings.warn(f"Row {row_num}: Missing source_url (recommended for verification)", UserWarning)
