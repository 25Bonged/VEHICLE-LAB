#!/usr/bin/env python3
import sys
sys.path.append('.')
try:
    import asammdf
    from pathlib import Path

    file_path = Path('uploads/20250528_1535_20250528_6237_PSALOGV2.mdf')
    print(f'Opening file: {file_path}')
    print(f'File exists: {file_path.exists()}')
    print(f'File size: {file_path.stat().st_size if file_path.exists() else 0} bytes')

    if file_path.exists():
        print('Opening MDF file...')
        mdf = asammdf.MDF(str(file_path))
        print(f'MDF opened successfully')
        print(f'Version: {mdf.version}')
        print(f'Number of channels in channels_db: {len(mdf.channels_db)}')
        print(f'Start time: {mdf.start_time}')
        print(f'Duration: {mdf.duration}')

        # Check if channels_db is empty
        if not mdf.channels_db:
            print('channels_db is empty')
        else:
            print('Sample channels from channels_db:')
            for i, (name, ch_info) in enumerate(list(mdf.channels_db.items())[:5]):
                print(f'  {i+1}. {name}')

        # Try to get some data
        if mdf.channels_db:
            first_channel = list(mdf.channels_db.keys())[0]
            print(f'\nTrying to read data from: {first_channel}')
            try:
                data = mdf.get(first_channel)
                print(f'Data shape: {data.shape if hasattr(data, "shape") else len(data)}')
                print(f'Data type: {type(data)}')
                print(f'Data length: {len(data) if hasattr(data, "__len__") else "N/A"}')
            except Exception as e:
                print(f'Error reading data: {e}')
    else:
        print('File does not exist')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
