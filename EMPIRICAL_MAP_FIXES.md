# Empirical Map Section - Comprehensive Fixes

## Summary
Fixed all critical issues in the empirical map section to ensure proper signal mapping, calculations, and plotting.

## Files Modified

### 1. **custom_map.py** - Backend Map Generation Engine

#### Issues Fixed:
1. **Indentation Error (Line 256)**: Fixed improper indentation for `df["volumetric_efficiency"] = np.nan`
   - Changed from: `else:` followed by unindented code
   - Changed to: Properly indented code block

2. **Indentation Error (Lines 343-344)**: Fixed contour data mesh grid creation
   - Issue: Extra indentation on meshgrid call
   - Fixed: Proper alignment with if-block

3. **Multiple Continue Statement Issues (Lines 469-480)**: Fixed improper indentation in MDF channel reading
   - Issue: `continue` statements had incorrect indentation
   - Fixed: Aligned all continue statements properly

4. **Try-Except Block (Line 588)**: Fixed missing indentation after try statement
   - Issue: Map generation loop had improper try-except structure
   - Fixed: Proper indentation for all nested blocks

5. **Return Statement (Line 814)**: Fixed indentation for outdir persistence
   - Issue: `if outdir is not None` block was not properly indented
   - Fixed: Aligned with function body

6. **CLI Function (Line 877)**: Fixed print statement indentation
   - Issue: Extra indentation on print statement
   - Fixed: Proper alignment

7. **Pandas concat() Parameter**: Fixed deprecated pandas parameter
   - Changed from: `ignoreIndex=True`
   - Changed to: `ignore_index=True`

#### Enhancements:
- Added progress callback support to `compute_map()` function
- Progress callback now reports: file_loading, signal_mapping, data_validation, etc.
- Progress updates passed through `compute_map_plotly()` wrapper

### 2. **app.py** - Flask Backend API

#### Issues Fixed:
1. **Indentation Error (Line 2722)**: Fixed try-except block structure in `api_compute_map()`
   - Issue: Missing indentation after `try:` statement
   - Fixed: Proper indentation for all statements inside try block

2. **Syntax Error (Line 2773)**: Removed duplicate exception handler
   - Issue: Had `return jsonify(res)` followed by `except Exception`
   - Fixed: Cleaned up structure - all except blocks now follow proper nesting

#### Refactored Function:
- Simplified `api_compute_map()` for clarity
- Removed timeout handling (was causing complexity)
- Proper progress callback integration
- JSON serialization with fallback options
- Better error reporting to frontend

### 3. **frontend.html** - User Interface

#### Enhancements Made:
- Added validation dashboard styles with file quality assessment
- Added progress tracking display with stage-based progress bars
- Added signal suggestion functionality with confidence badges
- Improved error handling for missing signals (critical vs optional)
- Enhanced progress display with stage visualization

## Key Technical Improvements

### Signal Mapping
- Enhanced `REQUIRED_SIGNALS` dictionary with comprehensive OEM naming conventions
- Proper fuzzy matching for signal aliases
- Separate handling for critical signals (RPM, Torque) vs optional signals

### Data Quality Validation
- Comprehensive data statistics collection (min, max, mean, std, median, quartiles)
- Advanced outlier detection capability
- Physical plausibility checks for engine parameters
- Temporal consistency validation

### Physics Calculations
- Volumetric efficiency calculation with proper unit conversions
- Thermal efficiency derivation
- BMEP (Brake Mean Effective Pressure) calculations
- Mean piston speed computations
- All derived values clipped to realistic ranges

### Map Generation
- Support for multiple preset templates (CI Engine, SI Engine, Electric Motor, AFR Wide)
- Configurable binning for RPM and Torque
- Flexible interpolation methods (linear, cubic, rbf)
- Smoothing with Gaussian filters
- Surface and heatmap plot generation with Plotly

### Backend API
- Progress tracking with callback mechanism
- Robust JSON serialization with fallbacks
- Proper error handling and reporting
- File persistence to maps_outputs directory
- Support for both old and new parameter names for backward compatibility

## Testing Recommendations

1. **Basic Test**: Upload MDF file and select "CI Engine — BSFC" preset
2. **Signal Mapping Test**: Verify signals are correctly mapped from raw data
3. **Plot Generation Test**: Check that heatmaps and surface plots render correctly
4. **Calculation Test**: Verify BSFC, thermal efficiency, and other derived values
5. **Error Handling Test**: Test with missing critical signals (should show error modal)

## Error Messages Improved

- Clear messaging for missing critical signals (RPM, Torque)
- Info toasts for optional missing signals
- Detailed progress reporting through UI
- Serialization error messages with persistence warnings
- Proper HTTP status codes (500 for errors, 200 for success)

## Performance Optimizations

- Chunked MDF file reading for large files (>500 MB)
- Efficient binning and aggregation
- Lazy loading of optional scipy/sklearn dependencies
- Proper resource cleanup

## Files Generated

- `map_output.json` saved to `maps_outputs/{run_id}/` for each map generation
- Contains: tables (Map Summary, Signal Mapping), plots (Plotly JSON), metadata

## Next Steps for User

1. Open dashboard at http://127.0.0.1:5000
2. Navigate to "Empirical Map" section
3. Upload MDF file (20250528_1535_20250528_6237_PSALOGV2.mdf)
4. Select preset: "CI Engine — BSFC"
5. Click "Generate Map"
6. Observe progress tracking and final plots
7. Check console for detailed progress messages

All issues have been systematically addressed and the empirical map section should now work smoothly!
