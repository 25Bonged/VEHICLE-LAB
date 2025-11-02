#!/usr/bin/env python3
"""
Test script to verify plot generation and output structure
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from custom_misfire import compute_misfire

test_file = Path("uploads/20250528_1535_20250528_6237_PSALOGV2.mdf")

print("🔍 Testing Plot Generation and Output Structure")
print("=" * 60)

result = compute_misfire([test_file], include_plots=True)

# Check plots
print(f"\n📈 Plot Analysis:")
plots = result.get('plots', {})
print(f"   Total plots generated: {len(plots)}")

for plot_name, plot_data in plots.items():
    print(f"\n   Plot: {plot_name}")
    if isinstance(plot_data, dict):
        if 'plotly_json' in plot_data:
            try:
                plot_json = json.loads(plot_data['plotly_json']) if isinstance(plot_data['plotly_json'], str) else plot_data['plotly_json']
                if 'data' in plot_json:
                    print(f"      ✓ Has data: {len(plot_json['data'])} traces")
                if 'layout' in plot_json:
                    print(f"      ✓ Has layout with title: {plot_json['layout'].get('title', {}).get('text', 'N/A')}")
                print(f"      ✓ Plotly JSON valid")
            except Exception as e:
                print(f"      ❌ Error parsing plotly_json: {e}")
        else:
            print(f"      ⚠️ Missing 'plotly_json' key")
    else:
        print(f"      ⚠️ Unexpected plot data format")

# Check table structure
print(f"\n📋 Table Structure Analysis:")
tables = result.get('tables', {})
print(f"   Total tables: {len(tables)}")

for table_name, table_data in tables.items():
    print(f"\n   Table: {table_name}")
    if isinstance(table_data, list):
        print(f"      Rows: {len(table_data)}")
        if len(table_data) > 0:
            print(f"      Sample columns: {list(table_data[0].keys())[:5]}")
            
            # Check for cylinder info in Misfire Events
            if table_name == 'Misfire Events' and len(table_data) > 0:
                events_with_cylinder = sum(1 for e in table_data if 'cylinder' in e)
                print(f"      Events with cylinder ID: {events_with_cylinder}/{len(table_data)}")
                events_with_confidence = sum(1 for e in table_data if 'confidence' in e)
                print(f"      Events with confidence: {events_with_confidence}/{len(table_data)}")
    else:
        print(f"      Type: {type(table_data).__name__}")

# Check meta information
print(f"\n📊 Meta Information:")
meta = result.get('meta', {})
print(f"   OK: {meta.get('ok', False)}")
print(f"   Files Processed: {meta.get('files_processed', 0)}")
print(f"   Total Misfires: {meta.get('total_misfires', 0)}")
print(f"   Detection Methods: {len(meta.get('detection_methods', []))}")
print(f"   OEM Features:")
for feature, enabled in meta.get('oem_features', {}).items():
    status = "✓" if enabled else "✗"
    print(f"      {status} {feature}")
print(f"   System Version: {meta.get('system_version', 'N/A')}")

print(f"\n✅ Plot and structure verification complete!")

