#!/usr/bin/env python3
"""
Script to collect DBC files from GitHub repositories
Collects all .dbc files from various automotive CAN bus repositories
"""

import os
import subprocess
import shutil
from pathlib import Path
import json
from typing import List, Dict
# Main repositories known to contain DBC files
REPOSITORIES = [
    "https://github.com/commaai/opendbc",
    "https://github.com/BogGyver/opendbc",
    "https://github.com/joshwardell/model3dbc",
    "https://github.com/vishrantgupta/DBC-CAN-Bus-Reader",
    "https://github.com/cantools/cantools",
]

# Output directory for collected DBC files
OUTPUT_DIR = Path("collected_dbc_files")
REPO_CACHE_DIR = Path("dbc_repo_cache")

def ensure_dir(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)

def clone_repo(repo_url: str, target_dir: Path) -> bool:
    """Clone a GitHub repository"""
    repo_name = repo_url.split("/")[-1]
    repo_path = target_dir / repo_name
    
    if repo_path.exists():
        print(f"  ✓ Repository {repo_name} already exists, updating...")
        try:
            subprocess.run(
                ["git", "pull"],
                cwd=repo_path,
                capture_output=True,
                check=False
            )
        except Exception as e:
            print(f"  ⚠ Could not update {repo_name}: {e}")
    else:
        print(f"  📥 Cloning {repo_name}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
                capture_output=True,
                check=True
            )
            print(f"  ✓ Cloned {repo_name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to clone {repo_name}: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            return False
        except FileNotFoundError:
            print(f"  ✗ Git not found. Please install git to clone repositories.")
            return False
    
    return True

def find_dbc_files(directory: Path) -> List[Path]:
    """Recursively find all .dbc files in a directory"""
    dbc_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-source dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.git']]
        
        for file in files:
            if file.endswith('.dbc'):
                dbc_files.append(Path(root) / file)
    return dbc_files

def copy_dbc_file(src: Path, dest_dir: Path, repo_name: str) -> Path:
    """Copy a DBC file to destination directory with organized naming"""
    # Create subdirectory structure: manufacturer/model/file.dbc
    relative_path = src.relative_to(REPO_CACHE_DIR / repo_name)
    
    # Create a safe filename with repo prefix
    safe_name = f"{repo_name}_{relative_path.name}"
    
    # If file is in a subdirectory, preserve some structure
    if len(relative_path.parts) > 1:
        # Use parent directory as part of name
        parent_name = relative_path.parent.name.replace('/', '_')
        safe_name = f"{repo_name}_{parent_name}_{relative_path.name}"
    
    dest_file = dest_dir / safe_name
    shutil.copy2(src, dest_file)
    return dest_file

# Note: GitHub API functionality removed to avoid requiring requests library
# All repositories are cloned directly using git

def main():
    """Main function to collect all DBC files"""
    print("=" * 70)
    print("DBC File Collection Script")
    print("=" * 70)
    
    ensure_dir(OUTPUT_DIR)
    ensure_dir(REPO_CACHE_DIR)
    
    all_dbc_files = []
    collection_summary = {}
    
    print(f"\n📁 Output directory: {OUTPUT_DIR.absolute()}")
    print(f"📦 Cache directory: {REPO_CACHE_DIR.absolute()}\n")
    
    # Process each repository
    for repo_url in REPOSITORIES:
        repo_owner = repo_url.split("/")[-2]
        repo_name = repo_url.split("/")[-1]
        print(f"\n🔍 Processing: {repo_owner}/{repo_name}")
        
        if clone_repo(repo_url, REPO_CACHE_DIR):
            repo_path = REPO_CACHE_DIR / repo_name
            dbc_files = find_dbc_files(repo_path)
            
            if dbc_files:
                print(f"  📄 Found {len(dbc_files)} DBC file(s)")
                
                repo_output_dir = OUTPUT_DIR / repo_name
                ensure_dir(repo_output_dir)
                
                copied_files = []
                for dbc_file in dbc_files:
                    dest_file = copy_dbc_file(dbc_file, repo_output_dir, repo_name)
                    copied_files.append(str(dest_file.relative_to(OUTPUT_DIR)))
                    all_dbc_files.append(dbc_file)
                
                collection_summary[repo_name] = {
                    "count": len(dbc_files),
                    "files": copied_files
                }
                print(f"  ✓ Copied {len(copied_files)} file(s) to {repo_output_dir}")
            else:
                print(f"  ℹ No DBC files found in {repo_name}")
                collection_summary[repo_name] = {"count": 0, "files": []}
        else:
            collection_summary[repo_name] = {"count": 0, "files": [], "error": "Failed to clone"}
    
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
    for repo, info in collection_summary.items():
        status = "✓" if info.get("count", 0) > 0 else "✗"
        print(f"  {status} {repo}: {info.get('count', 0)} file(s)")
    
    if total_count > 0:
        print(f"\n✅ Successfully collected {total_count} DBC file(s)!")
        print(f"\nTo view the files, check: {OUTPUT_DIR.absolute()}")
    else:
        print("\n⚠ No DBC files were found. Please check:")
        print("  1. Git is installed and accessible")
        print("  2. Internet connection is working")
        print("  3. Repository URLs are correct")

if __name__ == "__main__":
    main()

