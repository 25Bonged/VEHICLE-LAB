# Folder Upload Implementation Guide

## Quick Start: Add Folder Upload to Your Dashboard

This guide provides step-by-step instructions to add folder/directory upload functionality to your MDF Analytics Dashboard.

---

## Step 1: Update Frontend HTML

### Add Folder Upload Button

In `frontend.html`, find the upload section (around line 3288) and modify:

```html
<div id="upload-drop" tabindex="0" aria-label="Drag and drop MDF/MF4/Excel/CSV files or press Enter to browse">
  <strong>Drag & Drop Files or Folders</strong>
  <div style="opacity:.7">or click / Enter</div>
  
  <!-- Add folder upload button -->
  <div style="display:flex;gap:10px;justify-content:center;margin-top:12px">
    <button id="btn-upload-folder" type="button" 
            style="padding:8px 16px;background:var(--acc);border:none;border-radius:4px;cursor:pointer;color:white">
      📁 Upload Folder
    </button>
    <span class="small" style="opacity:.7">or drag files here</span>
  </div>
  
  <div id="upload-progress-bar"><div id="upload-progress"></div></div>
  <div style="display:flex;gap:10px;justify-content:center;margin-top:8px">
    <span class="badge" id="badge-filecount">Files: 0</span>
    <span class="badge" id="badge-totalsize">Total: 0</span>
    <span class="badge" id="badge-foldercount" style="display:none">Folders: 0</span>
  </div>
  <div id="drop-msg" class="small" style="opacity:.55">Ready</div>
  <label for="files" class="sr-only">Upload data files</label>
  <input id="files" type="file" accept=".mf4,.mdf,.csv,.xlsx,.xls" multiple style="display:none">
  
  <!-- Add hidden folder input -->
  <input id="folder-input" type="file" 
         webkitdirectory directory multiple
         accept=".mf4,.mdf,.csv,.xlsx,.xls"
         style="display:none">
</div>
```

---

## Step 2: Add JavaScript Functions

Add these functions to `frontend.html` in the JavaScript section (after the existing `handleUpload` function):

```javascript
// ====================== Folder Upload Support ======================

// Folder upload button handler
const folderInput = document.getElementById('folder-input');
const folderBtn = document.getElementById('btn-upload-folder');

if (folderBtn && folderInput) {
  folderBtn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    folderInput.click();
  });
  
  folderInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    
    // Show folder preview
    const folderInfo = analyzeFolderStructure(files);
    const preview = createFolderPreview(folderInfo);
    
    // Confirm upload
    const confirmed = await showFolderUploadDialog(folderInfo, preview);
    if (confirmed) {
      await handleFolderUpload(files);
    }
    
    // Reset input
    e.target.value = '';
  });
}

function analyzeFolderStructure(files) {
  const structure = {
    totalFiles: files.length,
    totalSize: 0,
    folders: new Set(),
    fileTypes: {},
    rootFiles: []
  };
  
  files.forEach(file => {
    structure.totalSize += file.size;
    
    // Track file types
    const ext = file.name.split('.').pop()?.toLowerCase() || 'unknown';
    structure.fileTypes[ext] = (structure.fileTypes[ext] || 0) + 1;
    
    if (file.webkitRelativePath) {
      const pathParts = file.webkitRelativePath.split('/');
      const folderPath = pathParts.slice(0, -1).join('/');
      
      if (folderPath) {
        structure.folders.add(folderPath);
      } else {
        structure.rootFiles.push(file.name);
      }
    } else {
      structure.rootFiles.push(file.name);
    }
  });
  
  return {
    ...structure,
    folders: Array.from(structure.folders),
    folderCount: structure.folders.size,
    totalSizeMB: (structure.totalSize / (1024 * 1024)).toFixed(2)
  };
}

function createFolderPreview(info) {
  let html = `
    <div style="max-height:300px;overflow-y:auto;border:1px solid var(--border);border-radius:4px;padding:12px;background:var(--bg-alt);margin:12px 0">
      <div style="margin-bottom:8px">
        <strong>📁 Folder Structure:</strong>
      </div>
      <div style="font-family:monospace;font-size:12px">
  `;
  
  if (info.folders.length > 0) {
    info.folders.slice(0, 10).forEach(folder => {
      html += `<div style="padding:2px 0">📁 ${folder}</div>`;
    });
    if (info.folders.length > 10) {
      html += `<div style="padding:2px 0;opacity:.7">... and ${info.folders.length - 10} more folders</div>`;
    }
  }
  
  html += `</div></div>`;
  return html;
}

function showFolderUploadDialog(info, preview) {
  return new Promise((resolve) => {
    const dialog = document.createElement('div');
    dialog.style.cssText = `
      position:fixed;top:0;left:0;right:0;bottom:0;
      background:rgba(0,0,0,0.7);z-index:10000;
      display:flex;align-items:center;justify-content:center;
    `;
    
    dialog.innerHTML = `
      <div style="background:var(--bg);padding:24px;border-radius:8px;max-width:600px;max-height:80vh;overflow-y:auto">
        <h3 style="margin-top:0">📁 Upload Folder</h3>
        <div style="margin:16px 0">
          <div><strong>Files:</strong> ${info.totalFiles}</div>
          <div><strong>Folders:</strong> ${info.folderCount}</div>
          <div><strong>Total Size:</strong> ${info.totalSizeMB} MB</div>
          <div style="margin-top:8px">
            <strong>File Types:</strong> ${Object.entries(info.fileTypes).map(([k,v]) => `${k}: ${v}`).join(', ')}
          </div>
        </div>
        ${preview}
        <div style="display:flex;gap:12px;margin-top:20px">
          <button id="dialog-confirm" style="flex:1;padding:10px;background:var(--acc);color:white;border:none;border-radius:4px;cursor:pointer">
            Upload
          </button>
          <button id="dialog-cancel" style="flex:1;padding:10px;background:var(--border);color:var(--text);border:none;border-radius:4px;cursor:pointer">
            Cancel
          </button>
        </div>
      </div>
    `;
    
    document.body.appendChild(dialog);
    
    dialog.querySelector('#dialog-confirm').addEventListener('click', () => {
      document.body.removeChild(dialog);
      resolve(true);
    });
    
    dialog.querySelector('#dialog-cancel').addEventListener('click', () => {
      document.body.removeChild(dialog);
      resolve(false);
    });
    
    dialog.addEventListener('click', (e) => {
      if (e.target === dialog) {
        document.body.removeChild(dialog);
        resolve(false);
      }
    });
  });
}

async function handleFolderUpload(files) {
  if (!files.length) return;
  
  visibleChannelCount = INITIAL_CHANNEL_BATCH;
  currentSearch = "";
  
  const searchInput = $('channel-search');
  if (searchInput) searchInput.value = "";
  
  setStatus('disc-status', `Uploading ${files.length} files from folder...`);
  setDropState('upload');
  setUploadProgress(5);
  
  // Update UI badges
  const folderCountBadge = $('badge-foldercount');
  if (folderCountBadge) {
    const folders = new Set();
    files.forEach(f => {
      if (f.webkitRelativePath) {
        const folderPath = f.webkitRelativePath.split('/').slice(0, -1).join('/');
        if (folderPath) folders.add(folderPath);
      }
    });
    folderCountBadge.textContent = `Folders: ${folders.size}`;
    folderCountBadge.style.display = 'inline-block';
  }
  
  const fd = new FormData();
  fd.append('mode', $('mode').value);
  fd.append('preserve_structure', 'true');  // Tell backend to preserve folder structure
  
  // Add files with their relative paths
  let uploadedCount = 0;
  const totalFiles = files.length;
  
  for (const file of files) {
    fd.append('files', file);
    if (file.webkitRelativePath) {
      fd.append(`path_${file.name}`, file.webkitRelativePath);
    }
    
    uploadedCount++;
    // Update progress
    const progress = 5 + (uploadedCount / totalFiles) * 85;
    setUploadProgress(Math.min(95, progress));
  }
  
  try {
    const data = await safeFetch('/smart_merge_upload', {
      method: 'POST',
      body: fd
    });
    
    if (!data.ok) throw new Error(data.error || 'Upload failed');
    
    log(`Uploaded ${files.length} file(s) from folder`);
    const list = data.channels || [];
    
    if (list.length) {
      if (list.length > PROGRESSIVE_CHANNEL_MAPPING_THRESHOLD) {
        mapChannelsProgressive(list);
      } else {
        state.channels = list.map(ch => ({
          id: ch.id || ch.name || ch.clean,
          clean: ch.clean || ch.name || ch.id,
          presence: ch.presence || '',
          present_count: ch.present_count ?? ch.occurrence ?? 0
        }));
        renderChannelTable();
        try {
          localStorage.setItem('mdf_channels', JSON.stringify(state.channels));
        } catch (e) {
          console.warn('Failed to store channels in localStorage:', e);
        }
        syncStateToPlayground();
      }
      state.selected.clear();
      const channelsContainer = $('channels-container');
      if (channelsContainer) channelsContainer.style.display = 'block';
      setStatus('disc-status', `Uploaded. Channels: ${list.length}`);
    } else {
      await refreshChannelsAfterUpload(true);
    }
    
    await refreshFileList();
    setUploadProgress(100);
    setTimeout(() => setUploadProgress(0), 800);
    
    // Refresh map UI
    loadUploadedFiles();
    initEmpiricalMap();
    
    if (state.fileCount > 0) {
      loadReportSection(state.currentReportSection, { force: true });
    }
    
    toast(`Successfully uploaded ${files.length} files from folder`, {type: 'success'});
    
  } catch (e) {
    toast('Folder upload failed: ' + e.message, {type: 'error'});
    setStatus('disc-status', 'Upload error: ' + e.message);
    setUploadProgress(0);
  } finally {
    setDropState('ready');
    // Hide folder count badge
    const folderCountBadge = $('badge-foldercount');
    if (folderCountBadge) folderCountBadge.style.display = 'none';
  }
}
```

---

## Step 3: Update Backend (app.py)

Modify the `smart_merge_upload` function in `app.py`:

```python
@app.post("/smart_merge_upload")
def smart_merge_upload():
    """
    Enhanced upload handler with folder structure preservation
    """
    mode = (request.args.get("mode") or request.form.get("mode") or "replace").lower()
    if mode not in ("append", "replace"):
        mode = "replace"
    
    preserve_structure = request.form.get("preserve_structure", "false").lower() == "true"
    
    if mode == "replace":
        try:
            purge_mdf_files_only()
            ACTIVE_FILES.clear(); CHANNELS_CACHE.clear(); SERIES_CACHE.clear()
        except Exception as e:
            app.logger.exception("Error during replace purge in smart_merge_upload: %s", e)
    
    uploaded = []
    folder_structure = {}
    
    try:
        def _write_file_stream(fobj, dest_path):
            """Existing implementation - keep as is"""
            tmp = dest_path.with_suffix(dest_path.suffix + ".part")
            total = 0
            with tmp.open("wb") as out:
                while True:
                    chunk = fobj.stream.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                        out.close()
                        try:
                            if tmp.exists():
                                tmp.unlink()
                        except Exception:
                            pass
                        raise ValueError(f"{fobj.filename} exceeds {MAX_UPLOAD_SIZE_MB}MB")
                    out.write(chunk)
            
            try:
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass
                os.replace(str(tmp), str(dest_path))
            except Exception:
                try:
                    shutil.move(str(tmp), str(dest_path))
                except Exception as e:
                    try:
                        if tmp.exists(): tmp.unlink()
                    except Exception:
                        pass
                    raise
        
        # Multi-file list (form field "files")
        if "files" in request.files:
            for f in request.files.getlist("files"):
                if not getattr(f, "filename", None):
                    continue
                
                try:
                    safe = _sanitize_filename(f.filename)
                    
                    # Handle folder structure preservation
                    relative_path_key = f"path_{f.filename}"
                    relative_path = request.form.get(relative_path_key, "")
                    
                    if preserve_structure and relative_path:
                        # Parse folder structure
                        path_parts = Path(relative_path).parts[:-1]  # Exclude filename
                        dest_dir = UPLOAD_DIR
                        
                        # Create folder structure
                        for part in path_parts:
                            safe_part = _sanitize_filename(part)
                            if safe_part:  # Skip empty parts
                                dest_dir = dest_dir / safe_part
                                dest_dir.mkdir(exist_ok=True)
                        
                        dest = (dest_dir / safe).resolve()
                        
                        # Validate destination is within UPLOAD_DIR
                        if not str(dest).startswith(str(UPLOAD_DIR.resolve())):
                            raise ValueError(f"Invalid path: {relative_path}")
                        
                        folder_structure[relative_path] = str(dest)
                    else:
                        dest = (UPLOAD_DIR / safe).resolve()
                    
                    _write_file_stream(f, dest)
                    uploaded.append(str(dest))
                    
                except ValueError as ve:
                    app.logger.warning("Upload rejected for %s: %s", getattr(f, "filename", "<unknown>"), ve)
                    # Continue processing other files
                    continue
                except Exception as e:
                    app.logger.exception("Failed to save uploaded file %s: %s", getattr(f, "filename", "<unknown>"), e)
                    continue
        
        # Single-file fallback (keep existing logic)
        if "file" in request.files and not uploaded:
            f = request.files["file"]
            if getattr(f, "filename", None):
                try:
                    safe = _sanitize_filename(f.filename)
                    dest = (UPLOAD_DIR / safe).resolve()
                    _write_file_stream(f, dest)
                    uploaded.append(str(dest))
                except ValueError as ve:
                    app.logger.warning("Upload rejected for %s: %s", getattr(f, "filename", "<unknown>"), ve)
                    return json_error(str(ve), 400)
                except Exception as e:
                    app.logger.exception("Failed to save single file upload %s: %s", getattr(f, "filename", "<unknown>"), e)
    
    except Exception as e:
        app.logger.exception("save_error in smart_merge_upload: %s", e)
        return json_error(f"save_error: {e.__class__.__name__}", 400, exc=e)
    
    if not uploaded:
        return json_error("no_files_received", 400)
    
    # Update ACTIVE_FILES
    try:
        if mode == "append":
            seen = set(ACTIVE_FILES)
            for p in uploaded:
                if p not in seen:
                    ACTIVE_FILES.append(p)
                    seen.add(p)
        else:
            ACTIVE_FILES[:] = uploaded
    except Exception as e:
        app.logger.exception("Failed to update ACTIVE_FILES after upload: %s", e)
    
    # Channel discovery
    try:
        files_objs = [Path(p) for p in ACTIVE_FILES if Path(p).exists()]
        channel_meta = discover_channels(files_objs) if files_objs else []
    except Exception as e:
        app.logger.exception("Failed to discover channels after upload: %s", e)
        channel_meta = []
    
    return safe_jsonify({
        "ok": True,
        "mode": mode,
        "active_files": ACTIVE_FILES,
        "added": uploaded,
        "channels": channel_meta,
        "channel_count": len(channel_meta),
        "folder_structure": folder_structure if preserve_structure else {},
        "preserved_structure": preserve_structure
    })
```

---

## Step 4: Test the Implementation

1. **Open the dashboard** in Chrome or Firefox
2. **Click "📁 Upload Folder"** button
3. **Select a folder** containing MDF/MF4/CSV files
4. **Review the preview dialog** showing folder structure
5. **Confirm upload**
6. **Verify files are uploaded** with folder structure preserved

### Browser Compatibility Notes

- ✅ **Chrome/Edge**: Full support
- ✅ **Firefox**: Full support  
- ⚠️ **Safari**: Limited support (may need fallback)
- ❌ **Mobile browsers**: Not supported (will gracefully degrade to file picker)

### Fallback for Safari

Add this detection:

```javascript
function supportsFolderUpload() {
  const input = document.createElement('input');
  input.type = 'file';
  return 'webkitdirectory' in input || 'directory' in input;
}

if (!supportsFolderUpload()) {
  const folderBtn = document.getElementById('btn-upload-folder');
  if (folderBtn) {
    folderBtn.style.display = 'none';
    // Show message: "Folder upload not supported in this browser"
  }
}
```

---

## Troubleshooting

### Issue: Files not preserving folder structure

**Solution:** Check that `preserve_structure` is set to `'true'` in FormData and backend is reading it correctly.

### Issue: Upload fails for large folders

**Solution:** Implement chunked upload or increase timeout settings.

### Issue: Browser doesn't show folder picker

**Solution:** Ensure you're using Chrome/Edge/Firefox. Safari has limited support.

---

## Next Steps

After implementing folder upload:

1. Add upload queue management (see main recommendations document)
2. Implement resumable uploads for large files
3. Add duplicate file detection
4. Enhance progress tracking with per-file progress

---

*Implementation Guide Version: 1.0*  
*Last Updated: 2024*

