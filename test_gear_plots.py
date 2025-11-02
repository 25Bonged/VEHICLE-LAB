#!/usr/bin/env python3
"""Test gear hunt plots generation"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from custom_gear import analyze_gear_hunting

test_file = Path("uploads/20250528_1535_20250528_6237_PSALOGV2.mdf")

print("🔍 TESTING GEAR HUNT PLOTS")
print("=" * 70)

if not test_file.exists():
    print(f"❌ Test file not found: {test_file}")
    sys.exit(1)

try:
    result = analyze_gear_hunting([test_file])
    
    print(f"\n📊 RESULT STRUCTURE:")
    print(f"   Has 'tables': {'tables' in result}")
    print(f"   Has 'plots': {'plots' in result}")
    print(f"   Has 'meta': {'meta' in result}")
    
    print(f"\n📈 PLOTS ANALYSIS:")
    plots = result.get('plots', {})
    print(f"   Total plots: {len(plots)}")
    
    for plot_name, plot_data in plots.items():
        print(f"\n   Plot: {plot_name}")
        print(f"      Type: {type(plot_data).__name__}")
        
        if isinstance(plot_data, dict):
            print(f"      Keys: {list(plot_data.keys())[:5]}")
            
            if 'plotly_json' in plot_data:
                plotly_data = plot_data['plotly_json']
                if isinstance(plotly_data, str):
                    try:
                        parsed = json.loads(plotly_data)
                        print(f"      ✓ plotly_json is string, parseable")
                        if 'data' in parsed:
                            print(f"      ✓ Has data: {len(parsed['data'])} traces")
                        if 'layout' in parsed:
                            print(f"      ✓ Has layout")
                    except:
                        print(f"      ✗ plotly_json is string but NOT parseable")
                else:
                    print(f"      ✓ plotly_json is object (not string)")
                    if isinstance(plotly_data, dict) and 'data' in plotly_data:
                        print(f"      ✓ Has data: {len(plotly_data['data'])} traces")
            else:
                print(f"      ✗ No 'plotly_json' key found")
        else:
            print(f"      ✗ Not a dict")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

