#!/usr/bin/env python3
"""
Test script for misfire detection with sample data
"""
import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from custom_misfire import compute_misfire
    print("✅ Successfully imported misfire detection module")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test with the MDF file
test_file = Path("uploads/20250528_1535_20250528_6237_PSALOGV2.mdf")

if not test_file.exists():
    print(f"❌ Test file not found: {test_file}")
    sys.exit(1)

print(f"\n🔍 Testing misfire detection on: {test_file.name}")
print("=" * 60)

try:
    result = compute_misfire([test_file], include_plots=True)
    
    print(f"\n📊 Results Summary:")
    print(f"   Status: {'✅ OK' if result['meta']['ok'] else '❌ Error'}")
    print(f"   Files Processed: {result['meta']['files_processed']}")
    print(f"   Total Misfires: {result['meta']['total_misfires']}")
    print(f"   Detection Methods: {len(result['meta']['detection_methods'])}")
    print(f"   OEM Features: {result['meta'].get('oem_features', {})}")
    
    # Check tables
    if 'tables' in result:
        print(f"\n📋 Tables Generated:")
        for table_name in result['tables'].keys():
            table_data = result['tables'][table_name]
            if isinstance(table_data, list):
                print(f"   - {table_name}: {len(table_data)} rows")
            else:
                print(f"   - {table_name}: {table_data}")
    
    # Check plots
    if 'plots' in result:
        print(f"\n📈 Plots Generated:")
        for plot_name in result['plots'].keys():
            print(f"   - {plot_name}")
    
    # Show file summary
    if 'tables' in result and 'File Summary' in result['tables']:
        print(f"\n📁 File Summary:")
        for file_summary in result['tables']['File Summary']:
            print(f"   File: {file_summary.get('file', 'Unknown')}")
            print(f"   Status: {file_summary.get('status', 'Unknown')}")
            print(f"   Misfires: {file_summary.get('misfires', 0)}")
            if 'signals_available' in file_summary:
                signals = file_summary['signals_available']
                print(f"   Signals Found:")
                for signal, available in signals.items():
                    status = "✅" if available else "❌"
                    print(f"      {status} {signal}")
            if 'per_cylinder_misfires' in file_summary:
                print(f"   Per-Cylinder Misfires: {file_summary['per_cylinder_misfires']}")
    
    # Show statistics
    if 'tables' in result and 'Statistics' in result['tables']:
        print(f"\n📊 Statistics:")
        for stat in result['tables']['Statistics']:
            print(f"   {stat.get('metric', 'Unknown')}: {stat.get('value', 'N/A')}")
    
    # Show sample events (first 5)
    if 'tables' in result and 'Misfire Events' in result['tables']:
        events = result['tables']['Misfire Events']
        print(f"\n🔍 Sample Misfire Events (first 5 of {len(events)}):")
        for i, event in enumerate(events[:5], 1):
            print(f"   Event {i}:")
            print(f"      Time: {event.get('time', 'N/A'):.2f}s")
            print(f"      RPM: {event.get('rpm', 'N/A'):.0f}")
            print(f"      Severity: {event.get('severity', 'Unknown')}")
            print(f"      Confidence: {event.get('confidence', 'N/A')}")
            if 'cylinder' in event:
                print(f"      Cylinder: {event['cylinder']}")
            if 'detection_methods' in event:
                print(f"      Methods: {event['detection_methods']}")
    
    print(f"\n✅ Test completed successfully!")
    print(f"   Total events: {len(result.get('tables', {}).get('Misfire Events', []))}")
    
except Exception as e:
    print(f"\n❌ Error during misfire detection: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

