#!/usr/bin/env python3
"""
Script to collect DBC files from GitHub repositories using ZIP downloads
Collects all .dbc files from various automotive CAN bus repositories
"""

import os
import zipfile
import shutil
from pathlib import Path
import json
from typing import List, Dict
import subprocess
import tempfile

# Main repositories known to contain DBC files
REPOSITORIES = [
    ("commaai", "opendbc"),
    ("BogGyver", "opendbc"),
    ("joshwardell", "model3dbc"),
    ("vishrantgupta", "DBC-CAN-Bus-Reader"),
    ("cantools", "cantools"),
    ("howerj", "dbcc"),
]

# Output directory for collected DBC files
OUTPUT_DIR = Path("collected_dbc_files")
TEMP_DIR = Path("dbc_temp_downloads")

def ensure_dir(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

def download_repo_zip(owner: str, repo: str, target_dir: Path) -> Path:
    """Download a GitHub repository as ZIP file"""
    zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
    repo_name = f"{owner}_{repo}"
    zip_file = target_dir / f"{repo_name}.zip"
    
    print(f"  📥 Downloading {owner}/{repo}...")
    
    # Try master branch first, then main
    for branch in ["master", "main"]:
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        try:
            result = subprocess.run(
                ["curl", "-L", "-f", "-o", str(zip_file), zip_url],
                capture_output=True,
                timeout=120,
                check=True
            )
            if zip_file.exists() and zip_file.stat().st_size > 0:
                print(f"  ✓ Downloaded {owner}/{repo}")
                return zip_file
        except subprocess.CalledProcessError:
            continue
        except Exception as e:
            print(f"  ⚠ Error downloading {owner}/{repo}: {e}")
    
    # If both branches failed, return None
    return None

def extract_zip(zip_file: Path, extract_dir: Path) -> Path:
    """Extract ZIP file and return the extracted directory"""
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Get the first directory name (usually repo-master or repo-main)
            namelist = zip_ref.namelist()
            if namelist:
                root_dir = namelist[0].split('/')[0]
                zip_ref.extractall(extract_dir)
                return extract_dir / root_dir
    except Exception as e:
        print(f"  ✗ Failed to extract {zip_file.name}: {e}")
    return None

def find_dbc_files(directory: Path) -> List[Path]:
    """Recursively find all .dbc files in a directory"""
    dbc_files = []
    if not directory.exists():
        return dbc_files
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.git', 'build', 'dist']]
        
        for file in files:
            if file.endswith('.dbc'):
                dbc_files.append(Path(root) / file)
    return dbc_files

def copy_dbc_file(src: Path, dest_dir: Path, repo_owner: str, repo_name: str) -> Path:
    """Copy a DBC file to destination directory with organized naming"""
    relative_path = src
    
    # Create a safe filename with repo prefix
    safe_name = f"{repo_owner}_{repo_name}_{relative_path.name}"
    
    # If file is in a subdirectory, preserve some structure info
    try:
        parts = src.parts
        # Try to extract meaningful directory info
        for i, part in enumerate(parts):
            if part.endswith('.dbc') or (i > 0 and 'dbc' in parts[i-1].lower()):
                # Include parent directory in name if it's meaningful
                if i > 0:
                    parent_name = parts[i-1].replace('/', '_').replace('\\', '_')
                    if parent_name and parent_name not in ['master', 'main', 'src', 'files']:
                        safe_name = f"{repo_owner}_{repo_name}_{parent_name}_{relative_path.name}"
                        break
    except:
        pass
    
    dest_file = dest_dir / safe_name
    shutil.copy2(src, dest_file)
    return dest_file

def main():
    """Main function to collect all DBC files"""
    print("=" * 70)
    print("DBC File Collection Script (ZIP Download Method)")
    print("=" * 70)
    
    ensure_dir(OUTPUT_DIR)
    ensure_dir(TEMP_DIR)
    
    all_dbc_files = []
    collection_summary = {}
    
    print(f"\n📁 Output directory: {OUTPUT_DIR.absolute()}")
    print(f"📦 Temp directory: {TEMP_DIR.absolute()}\n")
    
    # Process each repository
    for owner, repo in REPOSITORIES:
        repo_key = f"{owner}_{repo}"
        print(f"\n🔍 Processing: {owner}/{repo}")
        
        zip_file = download_repo_zip(owner, repo, TEMP_DIR)
        
        if zip_file and zip_file.exists():
            extract_dir = TEMP_DIR / repo_key
            ensure_dir(extract_dir)
            
            extracted_path = extract_zip(zip_file, extract_dir)
            
            if extracted_path and extracted_path.exists():
                dbc_files = find_dbc_files(extracted_path)
                
                if dbc_files:
                    print(f"  📄 Found {len(dbc_files)} DBC file(s)")
                    
                    repo_output_dir = OUTPUT_DIR / repo_key
                    ensure_dir(repo_output_dir)
                    
                    copied_files = []
                    for dbc_file in dbc_files:
                        try:
                            dest_file = copy_dbc_file(dbc_file, repo_output_dir, owner, repo)
                            copied_files.append(str(dest_file.relative_to(OUTPUT_DIR)))
                            all_dbc_files.append(dbc_file)
                        except Exception as e:
                            print(f"  ⚠ Failed to copy {dbc_file.name}: {e}")
                    
                    collection_summary[repo_key] = {
                        "count": len(dbc_files),
                        "files": copied_files,
                        "owner": owner,
                        "repo": repo
                    }
                    print(f"  ✓ Copied {len(copied_files)} file(s) to {repo_output_dir}")
                else:
                    print(f"  ℹ No DBC files found in {owner}/{repo}")
                    collection_summary[repo_key] = {
                        "count": 0,
                        "files": [],
                        "owner": owner,
                        "repo": repo
                    }
            else:
                print(f"  ✗ Failed to extract {repo_key}")
                collection_summary[repo_key] = {
                    "count": 0,
                    "files": [],
                    "error": "Failed to extract",
                    "owner": owner,
                    "repo": repo
                }
        else:
            print(f"  ✗ Failed to download {owner}/{repo}")
            collection_summary[repo_key] = {
                "count": 0,
                "files": [],
                "error": "Failed to download",
                "owner": owner,
                "repo": repo
            }
    
    # Create summary report
    summary_file = OUTPUT_DIR / "collection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_files": len(all_dbc_files),
            "repositories": collection_summary,
            "output_directory": str(OUTPUT_DIR.absolute())
        }, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("📊 Collection Summary")
    print("=" * 70)
    total_count = sum(s.get("count", 0) for s in collection_summary.values())
    print(f"Total DBC files collected: {total_count}")
    print(f"Total repositories processed: {len(REPOSITORIES)}")
    print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")
    print(f"Summary saved to: {summary_file}")
    
    print("\n📋 Repository Breakdown:")
    for repo_key, info in collection_summary.items():
        status = "✓" if info.get("count", 0) > 0 else "✗"
        count = info.get("count", 0)
        error = info.get("error", "")
        error_msg = f" ({error})" if error else ""
        print(f"  {status} {repo_key}: {count} file(s){error_msg}")
    
    if total_count > 0:
        print(f"\n✅ Successfully collected {total_count} DBC file(s)!")
        print(f"\nTo view the files, check: {OUTPUT_DIR.absolute()}")
        print(f"\nNote: Temporary files are in {TEMP_DIR.absolute()}")
        print(f"      You can delete this directory to free up space.")
    else:
        print("\n⚠ No DBC files were found. Please check:")
        print("  1. Internet connection is working")
        print("  2. Repository URLs are accessible")
        print("  3. curl is installed and working")

if __name__ == "__main__":
    main()

