#!/usr/bin/env python3
"""
Simple launcher for the MDF Analytics Dashboard
"""
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("🚀 Starting MDF Analytics Dashboard...")
    print("📁 Working directory:", os.getcwd())
    
    # Import and run the app
    from app import app
    
    print("✅ App imported successfully")
    print("🌐 Starting server on http://localhost:8000")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run the app
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=False,
        use_reloader=False
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Try installing dependencies: pip install flask pandas numpy plotly asammdf")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting dashboard: {e}")
    sys.exit(1)
