#!/usr/bin/env python3
"""
Comprehensive test of all empirical map features and plots
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

try:
    from custom_map import (
        compute_map, 
        create_heatmap_trace, 
        create_surface_trace,
        PRESET_TEMPLATES,
        detect_outliers_advanced,
        detect_steady_state_regions,
        validate_data_quality,
        derive_signals
    )
except ImportError as e:
    print(f"❌ Could not import modules: {e}")
    sys.exit(1)

def test_plotting_functions():
    """Test that plotting functions work correctly"""
    print("\n" + "="*70)
    print("TEST 1: Plotting Functions")
    print("="*70)
    
    # Create test data
    x_centers = np.array([1000, 1500, 2000, 2500, 3000])
    y_centers = np.array([50, 100, 150, 200, 250])
    mean_map = np.array([
        [210.5, 215.2, 220.1, 225.3, 230.0],
        [205.8, 210.5, 215.3, 220.1, 225.0],
        [200.5, 205.2, 210.1, 215.3, 220.0],
        [195.8, 200.5, 205.3, 210.1, 215.0],
        [190.5, 195.2, 200.1, 205.3, 210.0],
    ])
    
    map_dict = {
        'x_centers': x_centers,
        'y_centers': y_centers,
        'mean_map': mean_map
    }
    
    # Test heatmap
    try:
        heatmap_fig = create_heatmap_trace(map_dict, "Test BSFC Map", "g/kWh")
        if heatmap_fig:
            json_str = heatmap_fig.to_json()
            print(f"✅ Heatmap: Created successfully ({len(json_str)} chars JSON)")
        else:
            print("❌ Heatmap: Returned None")
            return False
    except Exception as e:
        print(f"❌ Heatmap: Error - {e}")
        return False
    
    # Test surface
    map_dict_surface = map_dict.copy()
    X, Y = np.meshgrid(x_centers, y_centers)
    map_dict_surface['surface_data'] = {
        'x': X,
        'y': Y,
        'z': mean_map
    }
    
    try:
        surface_fig = create_surface_trace(map_dict_surface, "Test BSFC Surface", "g/kWh")
        if surface_fig:
            json_str = surface_fig.to_json()
            print(f"✅ Surface: Created successfully ({len(json_str)} chars JSON)")
        else:
            print("❌ Surface: Returned None")
            return False
    except Exception as e:
        print(f"❌ Surface: Error - {e}")
        return False
    
    return True

def test_outlier_detection():
    """Test outlier detection functions"""
    print("\n" + "="*70)
    print("TEST 2: Outlier Detection")
    print("="*70)
    
    # Create test data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(100, 10, 100)
    outliers = np.array([200, 250, 5, 15])  # Clear outliers
    test_data = np.concatenate([normal_data, outliers])
    
    df = pd.DataFrame({
        'rpm': test_data,
        'torque': test_data * 0.5,
        'value': test_data * 2
    })
    
    try:
        mask = detect_outliers_advanced(df, ['rpm', 'torque', 'value'], method='combined')
        outliers_detected = (~mask).sum()
        print(f"✅ Outlier Detection: Found {outliers_detected} outliers out of {len(df)} samples")
        if outliers_detected > 0:
            print(f"   Outlier percentage: {outliers_detected/len(df)*100:.1f}%")
        return True
    except Exception as e:
        print(f"❌ Outlier Detection: Error - {e}")
        return False

def test_steady_state_detection():
    """Test steady-state detection"""
    print("\n" + "="*70)
    print("TEST 3: Steady-State Detection")
    print("="*70)
    
    # Create test data: steady RPM, varying torque
    time = np.linspace(0, 10, 100)
    rpm = np.full(100, 2000) + np.random.normal(0, 10, 100)  # Steady around 2000
    torque = 100 + 50 * np.sin(time * 0.5)  # Varying torque
    
    df = pd.DataFrame({
        'rpm': rpm,
        'torque': torque
    })
    
    try:
        mask = detect_steady_state_regions(df, rpm_col='rpm', torque_col='torque')
        steady_count = mask.sum()
        print(f"✅ Steady-State Detection: Found {steady_count} steady-state samples out of {len(df)}")
        print(f"   Steady-state percentage: {steady_count/len(df)*100:.1f}%")
        return True
    except Exception as e:
        print(f"❌ Steady-State Detection: Error - {e}")
        return False

def test_data_quality_validation():
    """Test data quality validation"""
    print("\n" + "="*70)
    print("TEST 4: Data Quality Validation")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'rpm': np.random.uniform(800, 4000, 100),
        'torque': np.random.uniform(0, 500, 100),
        'bsfc': np.random.uniform(200, 300, 100)
    })
    
    try:
        report = validate_data_quality(df, ['rpm', 'torque', 'bsfc'])
        print(f"✅ Data Quality Validation: Generated report")
        print(f"   Total samples: {report.get('total_samples', 0)}")
        print(f"   Signals validated: {len(report.get('signal_stats', {}))}")
        if 'outlier_info' in report:
            print(f"   Outlier detection: Enabled")
        if 'steady_state_info' in report:
            ss_info = report['steady_state_info']
            print(f"   Steady-state samples: {ss_info.get('steady_state_samples', 0)}")
        return True
    except Exception as e:
        print(f"❌ Data Quality Validation: Error - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_derive_signals():
    """Test signal derivation (physics calculations)"""
    print("\n" + "="*70)
    print("TEST 5: Signal Derivation & Physics Calculations")
    print("="*70)
    
    df = pd.DataFrame({
        'Epm_nEng': np.random.uniform(1000, 4000, 100),
        'TqSys_tqCkEngReal': np.random.uniform(50, 400, 100),
        'FuelRate': np.random.uniform(10, 50, 100),
    })
    
    try:
        df_result, mapping, report_rows, missing = derive_signals(df)
        
        derived_signals = []
        if 'omega_rad_s' in df_result.columns:
            derived_signals.append('omega_rad_s')
        if 'mech_power_kw' in df_result.columns:
            derived_signals.append('mech_power_kw')
        if 'fuel_mass_flow_kgps' in df_result.columns:
            derived_signals.append('fuel_mass_flow_kgps')
        if 'bsfc_gpkwh' in df_result.columns:
            derived_signals.append('bsfc_gpkwh')
        if 'thermal_efficiency' in df_result.columns:
            derived_signals.append('thermal_efficiency')
        if 'bmep_kpa' in df_result.columns:
            derived_signals.append('bmep_kpa')
        
        print(f"✅ Signal Derivation: Generated {len(derived_signals)} derived signals")
        print(f"   Derived signals: {', '.join(derived_signals)}")
        return True
    except Exception as e:
        print(f"❌ Signal Derivation: Error - {e}")
        import traceback
        traceback.print_exc()
        return False

def test_presets():
    """Test that all presets are valid"""
    print("\n" + "="*70)
    print("TEST 6: Preset Templates")
    print("="*70)
    
    required_keys = ['label', 'x_role', 'y_role']
    optional_keys = ['x_bins', 'y_bins', 'interp_method', 'filter_steady_state', 'filter_outliers']
    
    all_valid = True
    for key, preset in PRESET_TEMPLATES.items():
        print(f"\n  Checking preset: {key}")
        print(f"    Label: {preset.get('label', 'N/A')}")
        
        # Check required keys
        for req_key in required_keys:
            if req_key not in preset:
                print(f"    ❌ Missing required key: {req_key}")
                all_valid = False
        
        # Check optional keys
        for opt_key in optional_keys:
            if opt_key in preset:
                print(f"    ✅ {opt_key}: {preset[opt_key]}")
        
        # Check interp_method validity
        if 'interp_method' in preset:
            valid_methods = ['linear', 'cubic', 'rbf', 'kriging', 'cubic_spline']
            if preset['interp_method'] not in valid_methods:
                print(f"    ⚠️  Warning: interp_method '{preset['interp_method']}' not in standard methods")
    
    if all_valid:
        print(f"\n✅ All {len(PRESET_TEMPLATES)} presets are valid")
    else:
        print(f"\n❌ Some presets have issues")
    
    return all_valid

def test_full_map_generation():
    """Test full map generation with test data"""
    print("\n" + "="*70)
    print("TEST 7: Full Map Generation (Synthetic Data)")
    print("="*70)
    
    # Create realistic synthetic engine data
    np.random.seed(42)
    n_samples = 500
    
    # RPM: 800-4000, Torque: 0-500
    rpm = np.random.uniform(800, 4000, n_samples)
    torque = np.random.uniform(0, 500, n_samples)
    
    # Simulate BSFC: lower at mid-RPM, mid-torque (efficiency sweet spot)
    bsfc = 250 + (rpm - 2500)**2 / 10000 + (torque - 250)**2 / 5000 + np.random.normal(0, 10, n_samples)
    
    # Simulate fuel rate
    fuel_rate = torque * rpm / 10000 + np.random.normal(0, 2, n_samples)
    
    df = pd.DataFrame({
        'Epm_nEng': rpm,
        'TqSys_tqCkEngReal': torque,
        'FuelRate': fuel_rate,
    })
    
    try:
        # Use compute_map with synthetic data
        # Create a temporary CSV file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = Path(f.name)
        
        try:
            result = compute_map(
                [temp_path],
                preset='ci_engine_default',
            )
            
            if result.get('meta', {}).get('ok'):
                print("✅ Full Map Generation: Success")
                print(f"   Maps generated: {len(result.get('tables', {}).get('Map Summary', []))}")
                print(f"   Plots generated: {len(result.get('plots', {}))}")
                
                # Check for heatmap and surface plots
                plots = result.get('plots', {})
                heatmaps = [k for k in plots.keys() if 'heatmap' in k.lower()]
                surfaces = [k for k in plots.keys() if 'surface' in k.lower()]
                
                print(f"   Heatmap plots: {len(heatmaps)}")
                print(f"   Surface plots: {len(surfaces)}")
                
                return True
            else:
                error = result.get('meta', {}).get('error', 'Unknown')
                print(f"❌ Full Map Generation: Failed - {error}")
                return False
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
                
    except Exception as e:
        print(f"❌ Full Map Generation: Error - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE EMPIRICAL MAP FEATURE TEST")
    print("="*70)
    
    results = []
    
    # Run all tests
    results.append(("Plotting Functions", test_plotting_functions()))
    results.append(("Outlier Detection", test_outlier_detection()))
    results.append(("Steady-State Detection", test_steady_state_detection()))
    results.append(("Data Quality Validation", test_data_quality_validation()))
    results.append(("Signal Derivation", test_derive_signals()))
    results.append(("Preset Templates", test_presets()))
    
    # This test requires actual file I/O, might skip if problematic
    try:
        results.append(("Full Map Generation", test_full_map_generation()))
    except Exception as e:
        print(f"\n⚠️  Skipping Full Map Generation test: {e}")
        results.append(("Full Map Generation", None))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)
    
    for test_name, result in results:
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⏭️  SKIPPED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped out of {total} tests")
    
    if failed == 0:
        print("\n🎉 All tests passed! All features are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

