#!/usr/bin/env python3
"""
Comprehensive test of empirical map section
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from custom_map import compute_map
except ImportError as e:
    print(f"❌ Could not import compute_map: {e}")
    sys.exit(1)

test_file = Path("uploads/20250528_1535_20250528_6237_PSALOGV2.mdf")

print("🔍 COMPREHENSIVE EMPIRICAL MAP TEST")
print("=" * 70)

if not test_file.exists():
    print(f"❌ Test file not found: {test_file}")
    sys.exit(1)

print(f"\n📁 Testing with: {test_file.name}")
print(f"   File size: {test_file.stat().st_size / 1024 / 1024:.2f} MB\n")

try:
    result = compute_map([test_file])
    
    # Check basic structure
    print("📊 RESULT STRUCTURE CHECK:")
    print(f"   ✓ Has 'tables': {'tables' in result}")
    print(f"   ✓ Has 'plots': {'plots' in result}")
    print(f"   ✓ Has 'meta': {'meta' in result}")
    
    # Check meta
    print(f"\n📋 META INFORMATION:")
    meta = result.get('meta', {})
    print(f"   OK: {meta.get('ok', False)}")
    if 'error' in meta:
        print(f"   ❌ Error: {meta['error']}")
    else:
        print(f"   ✓ No errors in meta")
    
    # Check tables
    print(f"\n📊 TABLES ANALYSIS:")
    tables = result.get('tables', {})
    print(f"   Total tables: {len(tables)}")
    
    for table_name, table_data in tables.items():
        print(f"\n   Table: {table_name}")
        if isinstance(table_data, list):
            print(f"      Rows: {len(table_data)}")
            if len(table_data) > 0:
                print(f"      Columns: {list(table_data[0].keys())[:8]}")
        else:
            print(f"      Type: {type(table_data).__name__}")
            print(f"      Value: {str(table_data)[:100]}")
    
    # Check plots
    print(f"\n📈 PLOTS ANALYSIS:")
    plots = result.get('plots', {})
    print(f"   Total plots: {len(plots)}")
    
    plot_errors = []
    for plot_name, plot_data in plots.items():
        print(f"\n   Plot: {plot_name}")
        if isinstance(plot_data, dict):
            if 'plotly_json' in plot_data:
                try:
                    if isinstance(plot_data['plotly_json'], str):
                        plot_obj = json.loads(plot_data['plotly_json'])
                    else:
                        plot_obj = plot_data['plotly_json']
                    
                    if 'data' in plot_obj:
                        traces = len(plot_obj['data'])
                        print(f"      ✓ Traces: {traces}")
                    if 'layout' in plot_obj:
                        title = plot_obj['layout'].get('title', {})
                        if isinstance(title, dict):
                            title_text = title.get('text', 'N/A')
                        else:
                            title_text = str(title)
                        print(f"      ✓ Title: {title_text[:50]}")
                    print(f"      ✓ Plotly JSON valid")
                except Exception as e:
                    print(f"      ❌ ERROR: {e}")
                    plot_errors.append((plot_name, str(e)))
            else:
                print(f"      ❌ Missing 'plotly_json' key")
                plot_errors.append((plot_name, "Missing plotly_json"))
        else:
            print(f"      ❌ Unexpected format: {type(plot_data).__name__}")
            plot_errors.append((plot_name, f"Wrong format: {type(plot_data).__name__}"))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ TEST SUMMARY:")
    print(f"   Plots Generated: {len(plots)}")
    print(f"   Plot Errors: {len(plot_errors)}")
    print(f"   Tables: {len(tables)}")
    
    if plot_errors:
        print(f"\n❌ PLOT ERRORS FOUND:")
        for plot_name, error in plot_errors:
            print(f"   {plot_name}: {error}")
    else:
        print(f"   ✓ All plots valid")
    
    if 'error' in meta:
        print(f"\n❌ MAP GENERATION ERROR: {meta['error']}")
    else:
        print(f"   ✓ Map generation successful")
    
    print(f"\n✅ Empirical map section test complete!")

except Exception as e:
    print(f"\n❌ ERROR during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

