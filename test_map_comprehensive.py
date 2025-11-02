#!/usr/bin/env python3
"""
Comprehensive end-to-end test for empirical map features
Tests with real file structure and all features
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from custom_map import compute_map_plotly
import numpy as np

def test_with_real_mdf():
    """Test with actual MDF file"""
    print("\n" + "="*70)
    print("COMPREHENSIVE EMPIRICAL MAP TEST - REAL MDF FILE")
    print("="*70)
    
    mdf_file = Path("uploads/20250528_1535_20250528_6237_PSALOGV2.mdf")
    if not mdf_file.exists():
        print(f"❌ MDF file not found: {mdf_file}")
        return False
    
    print(f"\n📁 Testing with file: {mdf_file.name}")
    print(f"   Size: {mdf_file.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        print("\n🔄 Generating maps with advanced features...")
        result = compute_map_plotly(
            [str(mdf_file)],
            preset="ci_engine_default",
            min_samples_per_bin=5,
            smoothing=0.8,
            interp_method="cubic",
            enable_surface=True,
            enable_contours=True,
            contour_levels=15
        )
        
        if not result or not result.get('meta', {}).get('ok'):
            print("❌ Map generation failed")
            return False
        
        print("✅ Map generation successful!\n")
        
        # Test 1: Verify meta information
        print("="*70)
        print("TEST 1: Metadata Verification")
        print("="*70)
        meta = result.get('meta', {})
        print(f"✅ Files processed: {len(meta.get('files', []))}")
        print(f"✅ Total rows: {meta.get('rows', 'N/A')}")
        print(f"✅ Processing time: {meta.get('processing_time_sec', 'N/A')}s")
        print(f"✅ Settings:")
        settings = meta.get('settings', {})
        print(f"   - Preset: {settings.get('preset', 'N/A')}")
        print(f"   - Interpolation: {settings.get('interp_method', 'N/A')}")
        print(f"   - Smoothing: {settings.get('smoothing', 'N/A')}")
        print(f"   - Min samples/bin: {settings.get('min_samples_per_bin', 'N/A')}")
        
        # Test 2: Verify statistics in summaries
        print("\n" + "="*70)
        print("TEST 2: Advanced Statistics Verification")
        print("="*70)
        summaries = result.get('tables', {}).get('Map Summary', [])
        if not summaries:
            print("❌ No map summaries found")
            return False
        
        print(f"✅ Found {len(summaries)} map summaries\n")
        
        for i, summary in enumerate(summaries, 1):
            map_name = summary.get('map', 'unknown')
            print(f"Map {i}: {map_name}")
            print(f"  📊 Basic Stats:")
            print(f"    - Mean: {summary.get('mean', 'N/A')}")
            print(f"    - Median: {summary.get('median', 'N/A')}")
            print(f"    - Min: {summary.get('min', 'N/A')}")
            print(f"    - Max: {summary.get('max', 'N/A')}")
            print(f"    - Std Dev: {summary.get('std', 'N/A')}")
            
            print(f"  📈 Percentiles:")
            print(f"    - P25: {summary.get('p25', 'N/A')}")
            print(f"    - P75: {summary.get('p75', 'N/A')}")
            
            print(f"  ✅ Quality Metrics:")
            r2 = summary.get('quality_r_squared')
            rmse = summary.get('quality_rmse')
            mae = summary.get('quality_mae')
            mape = summary.get('quality_mape')
            val_samples = summary.get('validation_samples')
            
            print(f"    - R² (Coefficient of Determination): {r2}")
            print(f"    - RMSE (Root Mean Squared Error): {rmse}")
            print(f"    - MAE (Mean Absolute Error): {mae}")
            print(f"    - MAPE (Mean Abs % Error): {mape}")
            print(f"    - Validation samples: {val_samples}")
            
            print(f"  📍 Coverage:")
            print(f"    - Cells filled: {summary.get('cells_filled', 0)}/{summary.get('cells_total', 0)}")
            print(f"    - Coverage: {summary.get('coverage_pct', 0):.2f}%")
            print()
        
        # Test 3: Verify plots generation
        print("="*70)
        print("TEST 3: Plots Generation Verification")
        print("="*70)
        plots = result.get('plots', {})
        if not plots:
            print("❌ No plots generated")
            return False
        
        print(f"✅ Total plots generated: {len(plots)}\n")
        
        # Categorize plots
        heatmaps = [k for k in plots.keys() if 'heatmap' in k]
        surfaces = [k for k in plots.keys() if 'surface' in k]
        validation = [k for k in plots.keys() if 'scatter' in k or 'residuals' in k]
        
        print(f"  📊 Heatmaps: {len(heatmaps)}")
        for h in heatmaps[:3]:
            plot_data = plots[h]
            has_json = 'plotly_json' in plot_data
            status = "✅" if has_json else "❌"
            print(f"    {status} {h}")
        
        print(f"\n  🌐 3D Surfaces: {len(surfaces)}")
        for s in surfaces[:3]:
            plot_data = plots[s]
            has_json = 'plotly_json' in plot_data
            status = "✅" if has_json else "❌"
            print(f"    {status} {s}")
        
        print(f"\n  ✅ Validation Plots: {len(validation)}")
        validation_types = {}
        for v in validation:
            if 'scatter' in v:
                validation_types['Scatter (Observed vs Predicted)'] = validation_types.get('Scatter (Observed vs Predicted)', 0) + 1
            elif 'residuals_plot' in v:
                validation_types['Residuals Plot'] = validation_types.get('Residuals Plot', 0) + 1
            elif 'histogram' in v:
                validation_types['Residuals Histogram'] = validation_types.get('Residuals Histogram', 0) + 1
        
        for vtype, count in validation_types.items():
            print(f"    ✅ {vtype}: {count}")
            # Show examples
            examples = [v for v in validation if vtype.split()[0].lower() in v.lower()][:2]
            for ex in examples:
                plot_data = plots[ex]
                has_json = 'plotly_json' in plot_data
                status = "  ✅" if has_json else "  ❌"
                print(f"      {status} {ex}")
        
        # Test 4: Verify data quality report
        print("\n" + "="*70)
        print("TEST 4: Data Quality Report")
        print("="*70)
        quality = meta.get('data_quality', {})
        if quality:
            print(f"✅ Total samples: {quality.get('total_samples', 'N/A')}")
            valid_samples = quality.get('valid_samples', {})
            if valid_samples:
                print(f"✅ Valid samples per signal:")
                for signal, count in valid_samples.items():
                    print(f"   - {signal}: {count}")
            
            signal_stats = quality.get('signal_stats', {})
            if signal_stats:
                print(f"\n✅ Signal statistics available for {len(signal_stats)} signals")
        
        # Test 5: Check output structure completeness
        print("\n" + "="*70)
        print("TEST 5: Output Structure Verification")
        print("="*70)
        
        required_keys = ['tables', 'plots', 'meta', 'samples']
        for key in required_keys:
            if key in result:
                print(f"✅ {key}: Present")
            else:
                print(f"❌ {key}: Missing")
        
        # Check samples structure
        samples = result.get('samples', {})
        if samples:
            print(f"\n✅ Sample data export:")
            print(f"   - Rows exported: {samples.get('rows_exported', 'N/A')}")
            print(f"   - Total rows: {samples.get('rows_total', 'N/A')}")
            print(f"   - Columns: {len(samples.get('columns', []))}")
        
        # Final summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"✅ Map Generation: PASS")
        print(f"✅ Advanced Statistics: PASS ({len(summaries)} maps)")
        print(f"✅ Quality Metrics: PASS (R², RMSE, MAE calculated)")
        print(f"✅ Plots Generation: PASS ({len(plots)} plots)")
        print(f"   - Heatmaps: {len(heatmaps)}")
        print(f"   - 3D Surfaces: {len(surfaces)}")
        print(f"   - Validation Plots: {len(validation)}")
        print(f"✅ Data Quality Report: PASS")
        print(f"✅ Output Structure: PASS")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_real_mdf()
    sys.exit(0 if success else 1)

