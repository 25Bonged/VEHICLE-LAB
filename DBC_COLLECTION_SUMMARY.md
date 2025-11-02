# DBC File Collection Summary

**Date Collected:** November 1, 2025  
**Total DBC Files:** 341 files  
**Source Repositories:** 6 GitHub repositories

---

## Collection Statistics

| Repository | Owner | DBC Files | Status |
|------------|-------|-----------|--------|
| **commaai/opendbc** | commaai | 111 files | ✅ Success |
| **BogGyver/opendbc** | BogGyver | 143 files | ✅ Success |
| **joshwardell/model3dbc** | joshwardell | 1 file | ✅ Success |
| **vishrantgupta/DBC-CAN-Bus-Reader** | vishrantgupta | 0 files | ℹ️ No DBC files (parser tool) |
| **cantools/cantools** | cantools | 79 files | ✅ Success |
| **howerj/dbcc** | howerj | 7 files | ✅ Success |

---

## Vehicle Manufacturers Covered

The collected DBC files cover a wide range of vehicle manufacturers and models:

### Toyota/Lexus
- Prius, Corolla, Camry, Highlander, RAV4, Sienna, Avalon, iQ
- Lexus: RX, NX, IS, CT, GS models
- Various powertrain configurations (hybrid, standard)

### Honda/Acura
- Civic, CR-V, Accord, Odyssey, Pilot, Fit, Clarity, Insight, HR-V, Ridgeline
- Acura: ILX, RDX
- Multiple year ranges (2016-2019)

### Subaru
- Forester, Outback, Global models
- 2015-2020 model years
- Hybrid variants

### Ford/Lincoln
- Fusion, CGEA platform
- Lincoln base models
- ADAS and powertrain variants

### GM (General Motors)
- Global A platform (powertrain, chassis, lowspeed, object, high voltage management)
- Cadillac CT6
- Opel Omega

### Chrysler/Stellantis
- Pacifica, Ram (DT, HD)
- CUSW, Giorgio platforms
- Hybrid variants

### Hyundai/Kia
- Santafe, i30, Palisade, CanFD variants
- Kia generic files
- Multiple CAN buses (CCAN, MCAN)

### Nissan/Infiniti
- Leaf, Xterra, X-Trail
- Common DBC files

### Volkswagen/Audi
- MQB, MQBEVO, MEV, PQ platforms
- Golf MK4
- Various model years

### Tesla
- Model 3 (vehicle, powertrain, party CAN)
- Tesla CAN, radar, powertrain files

### BMW
- E9x/E8x platform

### Mercedes-Benz
- E350 2010

### Volvo
- V40 2017, V60 2015 (powertrain)

### Mazda
- RX8, 2017, 2019, 3 2019 models
- Radar files

### Rivian
- Primary actuator, park assist CAN

### Other Manufacturers
- GWM Haval H6 PHEV 2024
- Hongqi HS5
- Luxgen S5 2015
- PSA/Stellantis AEE2010 R3

---

## Common/Specialized DBC Files

- **Bosch**: 2018, 2020, radar ACC, standstill variants
- **ESR**: EyeSight system files
- **Steering**: Sensors A/B, control A/B
- **Gearbox**: Common gearbox files
- **LKAS HUD**: 5-byte and 8-byte variants
- **Nidec**: SCM group A/B, common files
- **Dual CAN**: Nidec 2018 variants
- **Community**: Community-contributed files

---

## File Locations

**Collected Files:** `/Users/chayan/Documents/project_25/backend_mdf/collected_dbc_files/`

The files are organized by repository:
- `commaai_opendbc/` - 111 files
- `BogGyver_opendbc/` - 143 files
- `joshwardell_model3dbc/` - 1 file
- `cantools_cantools/` - 79 files
- `howerj_dbcc/` - 7 files

**Summary JSON:** `collected_dbc_files/collection_summary.json`

---

## Notable Features

1. **Comprehensive Coverage**: Files from major automotive manufacturers
2. **Multiple Platforms**: Different CAN bus platforms (standard CAN, CAN-FD)
3. **Year Ranges**: Coverage from 2001 to 2024
4. **Powertrain Variants**: Standard, hybrid, electric configurations
5. **Specialized Systems**: ADAS, radar, steering, gearbox specific files
6. **Test Files**: cantools repository includes test/example DBC files

---

## Usage Notes

- All files are standard DBC format (Vector CANoe/CANalyzer compatible)
- Files can be used with CAN bus analysis tools
- Some files may require specific CAN bus configurations (CAN vs CAN-FD)
- Test files in cantools repository are for parser testing and may not represent real vehicles

---

## Source Repositories

1. **commaai/opendbc**: https://github.com/commaai/opendbc
   - Comprehensive collection of open DBC files for various vehicles
   - Maintained by comma.ai for their self-driving platform

2. **BogGyver/opendbc**: https://github.com/BogGyver/opendbc
   - Fork of opendbc with additional files and variations

3. **joshwardell/model3dbc**: https://github.com/joshwardell/model3dbc
   - Tesla Model 3 and Model Y CAN message definitions

4. **cantools/cantools**: https://github.com/cantools/cantools
   - Python library for CAN bus tools with test DBC files

5. **howerj/dbcc**: https://github.com/howerj/dbcc
   - DBC compiler with example/test files

---

## Collection Script

The collection was performed using `collect_dbc_files_v2.py`, which:
- Downloads repositories as ZIP files
- Extracts and searches for .dbc files
- Organizes files by repository
- Creates a summary JSON report

To re-run the collection:
```bash
python3 collect_dbc_files_v2.py
```

---

**Status:** ✅ Collection Complete - 341 DBC files successfully collected

