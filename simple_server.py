#!/usr/bin/env python3
"""
Simple test server to verify Flask is working
"""
from flask import Flask, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MDF Analytics Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .status { color: green; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 MDF Analytics Dashboard</h1>
            <p class="status">✅ Dashboard is running successfully!</p>
            <p>This is a simple test server to verify Flask is working.</p>
            <p>To access the full dashboard, run the main app.py file.</p>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Starting simple test server...")
    print("🌐 Server will be available at: http://localhost:8000")
    app.run(host='127.0.0.1', port=8000, debug=False)
