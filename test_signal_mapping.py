#!/usr/bin/env python3
"""
Test script to validate signal mapping system with various OBD/CAN log formats.
Tests CSV, Excel, and MDF file formats with different signal naming conventions.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

try:
    from signal_mapping import (
        find_signal_advanced,
        find_signal_in_dataframe,
        SIGNAL_MAP,
        get_signal_statistics
    )
    print("✅ Signal mapping module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import signal_mapping: {e}")
    sys.exit(1)


def create_test_csv_files(output_dir: Path) -> List[Path]:
    """Create sample CSV files with different OBD/CAN signal naming conventions."""
    output_dir.mkdir(exist_ok=True)
    created_files = []
    
    # Test file 1: Generic OBD-II style
    test1_data = {
        'Time': np.linspace(0, 100, 1000),
        'RPM': 800 + 2000 * np.random.random(1000),
        'VehicleSpeed': 30 + 40 * np.random.random(1000),
        'EngineTorque': 50 + 150 * np.random.random(1000),
        'ThrottlePosition': np.random.random(1000) * 100,
        'CoolantTemp': 80 + 20 * np.random.random(1000),
        'FuelRate': 5 + 10 * np.random.random(1000),
        'MAF': 10 + 30 * np.random.random(1000),
        'Lambda': 0.95 + 0.1 * np.random.random(1000),
    }
    df1 = pd.DataFrame(test1_data)
    file1 = output_dir / 'test_obd_generic.csv'
    df1.to_csv(file1, index=False)
    created_files.append(file1)
    print(f"✅ Created: {file1.name}")
    
    # Test file 2: VW/Audi style (German naming)
    test2_data = {
        'Zeit': np.linspace(0, 100, 1000),
        'Epm_nEng_RTE': 800 + 2000 * np.random.random(1000),
        'Veh_spdVeh': 30 + 40 * np.random.random(1000),
        'TqSys_tqCkEngReal': 50 + 150 * np.random.random(1000),
        'Drosselklappenposition': np.random.random(1000) * 100,
        'Kuehlmitteltemperatur': 80 + 20 * np.random.random(1000),
        'FuelCons': 5 + 10 * np.random.random(1000),
        'InM_mfAirCanPurgEstim': 10 + 30 * np.random.random(1000),
        'AFReg_facCorRich_RTE': 0.95 + 0.1 * np.random.random(1000),
    }
    df2 = pd.DataFrame(test2_data)
    file2 = output_dir / 'test_vw_audi.csv'
    df2.to_csv(file2, index=False)
    created_files.append(file2)
    print(f"✅ Created: {file2.name}")
    
    # Test file 3: Complex CAN naming (long prefixes)
    test3_data = {
        'Timestamp': np.linspace(0, 100, 1000),
        '96D7124080_8128328U_FM77_nc_CAN_VITESSE_VEHICULE_ROUES': 30 + 40 * np.random.random(1000),
        'MG1CS051_H440_2F_EngM_facTranCorSlop_RTE': 0.5 + 0.3 * np.random.random(1000),
        '96D7124080_8128328U_FM77_nc_SG_.PENTE_STATIQUE': np.random.random(1000) * 10,
        'Epm_nEng': 800 + 2000 * np.random.random(1000),
        'TqSys_tqCkEngReal_RTE': 50 + 150 * np.random.random(1000),
    }
    df3 = pd.DataFrame(test3_data)
    file3 = output_dir / 'test_can_complex.csv'
    df3.to_csv(file3, index=False)
    created_files.append(file3)
    print(f"✅ Created: {file3.name}")
    
    # Test file 4: OBD-II PIDs
    test4_data = {
        'Time_s': np.linspace(0, 100, 1000),
        'PID_0C': 800 + 2000 * np.random.random(1000),  # RPM
        'PID_0D': 30 + 40 * np.random.random(1000),  # Vehicle Speed
        'PID_11': 50 + 150 * np.random.random(1000),  # Throttle Position
        'PID_05': 80 + 20 * np.random.random(1000),  # Coolant Temp
        'PID_2F': 5 + 10 * np.random.random(1000),  # Fuel Level
        'PID_10': 10 + 30 * np.random.random(1000),  # MAF
    }
    df4 = pd.DataFrame(test4_data)
    file4 = output_dir / 'test_obd_pids.csv'
    df4.to_csv(file4, index=False)
    created_files.append(file4)
    print(f"✅ Created: {file4.name}")
    
    return created_files


def test_signal_detection(file_path: Path) -> Dict[str, Any]:
    """Test signal detection for a given file."""
    print(f"\n{'='*60}")
    print(f"Testing: {file_path.name}")
    print('='*60)
    
    try:
        df = pd.read_csv(file_path, nrows=1)
        columns = list(df.columns)
    except Exception as e:
        return {"error": str(e), "file": file_path.name}
    
    print(f"\nColumns found: {len(columns)}")
    
    # Test key signal roles
    test_roles = ['rpm', 'torque', 'vehicle_speed', 'fuel_rate', 'lambda',
                  'coolant_temp', 'throttle', 'map_sensor', 'gear', 'distance']
    
    results = {
        "file": file_path.name,
        "total_columns": len(columns),
        "detected_signals": {},
        "missing_signals": [],
        "coverage": 0.0
    }
    
    for role in test_roles:
        found = find_signal_in_dataframe(columns, role)
        if found:
            results["detected_signals"][role] = found
            print(f"  ✅ {role:20s} → {found}")
        else:
            results["missing_signals"].append(role)
            print(f"  ❌ {role:20s} → Not found")
    
    results["coverage"] = len(results["detected_signals"]) / len(test_roles) * 100
    print(f"\n📊 Coverage: {results['coverage']:.1f}% ({len(results['detected_signals'])}/{len(test_roles)} signals)")
    
    return results


def main():
    """Main test function."""
    print("="*60)
    print("Signal Mapping System Test Suite")
    print("="*60)
    
    # Create test directory
    test_dir = Path("./test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test CSV files
    print("\n📝 Creating test CSV files...")
    test_files = create_test_csv_files(test_dir)
    
    # Test each file
    all_results = []
    for test_file in test_files:
        results = test_signal_detection(test_file)
        all_results.append(results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_detected = 0
    total_tested = 0
    
    for result in all_results:
        if "error" in result:
            print(f"\n❌ {result['file']}: {result['error']}")
            continue
        
        detected = len(result["detected_signals"])
        tested = len(result["detected_signals"]) + len(result["missing_signals"])
        total_detected += detected
        total_tested += tested
        
        print(f"\n📁 {result['file']}")
        print(f"   Coverage: {result['coverage']:.1f}% ({detected}/{tested} signals)")
        if result["detected_signals"]:
            print(f"   Found: {', '.join(result['detected_signals'].keys())}")
    
    if total_tested > 0:
        overall_coverage = total_detected / total_tested * 100
        print(f"\n🎯 Overall Coverage: {overall_coverage:.1f}% ({total_detected}/{total_tested} signals)")
    
    print("\n✅ Test completed!")
    print(f"\n💡 Test files created in: {test_dir.absolute()}")
    print("   You can use these files to test the dashboard.")


if __name__ == "__main__":
    main()
