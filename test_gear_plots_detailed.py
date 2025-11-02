#!/usr/bin/env python3
"""Detailed test of gear hunt plot structure"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from custom_gear import analyze_gear_hunting

test_file = Path("uploads/20250528_1535_20250528_6237_PSALOGV2.mdf")

print("🔍 DETAILED GEAR HUNT PLOT TEST")
print("=" * 70)

if not test_file.exists():
    print(f"❌ Test file not found: {test_file}")
    sys.exit(1)

try:
    result = analyze_gear_hunting([test_file])
    
    print(f"\n📊 RESULT STRUCTURE:")
    print(f"   Has 'plots': {'plots' in result}")
    print(f"   Plot count: {len(result.get('plots', {}))}")
    
    print(f"\n📈 PLOT DETAILS:")
    plots = result.get('plots', {})
    
    for plot_name, plot_data in plots.items():
        print(f"\n   Plot: {plot_name}")
        print(f"      Type: {type(plot_data).__name__}")
        
        if isinstance(plot_data, dict):
            print(f"      Keys: {list(plot_data.keys())}")
            
            if 'plotly_json' in plot_data:
                plotly_str = plot_data['plotly_json']
                print(f"      plotly_json type: {type(plotly_str).__name__}")
                print(f"      plotly_json length: {len(str(plotly_str))}")
                
                try:
                    if isinstance(plotly_str, str):
                        plotly_obj = json.loads(plotly_str)
                    else:
                        plotly_obj = plotly_str
                    
                    print(f"      ✓ Parsed successfully")
                    print(f"      Has 'data': {'data' in plotly_obj}")
                    print(f"      Has 'layout': {'layout' in plotly_obj}")
                    
                    if 'data' in plotly_obj:
                        data = plotly_obj['data']
                        print(f"      Data type: {type(data).__name__}")
                        if isinstance(data, list):
                            print(f"      Traces count: {len(data)}")
                            for i, trace in enumerate(data[:3]):  # First 3 traces
                                print(f"         Trace {i}: type={trace.get('type', 'unknown')}, has_x={bool(trace.get('x'))}, has_y={bool(trace.get('y'))}")
                    
                    if 'layout' in plotly_obj:
                        layout = plotly_obj['layout']
                        print(f"      Layout has title: {'title' in layout}")
                        if 'title' in layout:
                            title = layout['title']
                            if isinstance(title, dict):
                                print(f"      Title text: {title.get('text', 'N/A')[:50]}")
                            else:
                                print(f"      Title: {str(title)[:50]}")
                        
                except json.JSONDecodeError as e:
                    print(f"      ✗ JSON decode error: {e}")
                    print(f"      First 200 chars: {str(plotly_str)[:200]}")
                except Exception as e:
                    print(f"      ✗ Parse error: {e}")
            else:
                print(f"      ✗ No 'plotly_json' key")
        else:
            print(f"      ✗ Not a dict: {type(plot_data)}")
            print(f"      Value: {str(plot_data)[:100]}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

