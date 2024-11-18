from PIL import Image
import os
from pathlib import Path
import cairosvg
import io
import shutil
import sys

def refresh_windows_icons():
    """Force Windows to refresh icon cache"""
    try:
        # Kill explorer
        os.system('taskkill /f /im explorer.exe')
        
        # Clear icon cache
        cache_paths = [
            '%LOCALAPPDATA%\\IconCache.db',
            '%LOCALAPPDATA%\\Microsoft\\Windows\\Explorer\\iconcache*',
            '%LOCALAPPDATA%\\Microsoft\\Windows\\Explorer\\thumbcache*'
        ]
        for path in cache_paths:
            os.system(f'del /f /s /q {path}')
        
        # Clear DNS cache (sometimes helps with icon refresh)
        os.system('ipconfig /flushdns')
        
        # Restart explorer
        os.system('start explorer.exe')
        
        # Additional icon refresh commands
        os.system('ie4uinit.exe -show')
        os.system('ie4uinit.exe -ClearIconCache')
    except Exception as e:
        print(f"Warning: Could not refresh icons: {e}")

def set_drive_attributes(drive_path):
    """Set attributes for drive icon files"""
    try:
        drive_letter = str(drive_path)[0].upper()
        
        # Set file attributes
        os.system(f'attrib +s +h "{drive_path}/.VolumeIcon.ico"')
        os.system(f'attrib +s +h "{drive_path}/autorun.inf"')
        
        # Add registry keys for drive icon
        reg_commands = [
            # Enable AutoRun
            'reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer" /v "NoDriveTypeAutoRun" /t REG_DWORD /d 0 /f',
            
            # Set drive icon in multiple locations
            f'reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\DriveIcons\\{drive_letter}" /ve /d "" /f',
            f'reg add "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\DriveIcons\\{drive_letter}\\DefaultIcon" /ve /d "{drive_path}\\.VolumeIcon.ico" /f',
            f'reg add "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\DriveIcons\\{drive_letter}" /ve /d "" /f',
            f'reg add "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\DriveIcons\\{drive_letter}\\DefaultIcon" /ve /d "{drive_path}\\.VolumeIcon.ico" /f',
            
            # Additional shell icon settings
            f'reg add "HKCR\\Drive\\shell\\{drive_letter}" /v "Icon" /d "{drive_path}\\.VolumeIcon.ico" /f'
        ]
        
        for cmd in reg_commands:
            os.system(cmd)
            
    except Exception as e:
        print(f"Warning: Could not set all drive attributes: {e}")

def set_folder_attributes(folder_path):
    """Set attributes for folder icon files"""
    os.system(f'attrib +s +h "{folder_path}/desktop.ini"')
    os.system(f'attrib +s +h "{folder_path}/folder.ico"')
    os.system(f'attrib +r "{folder_path}"')

def safe_remove(path):
    """Safely remove a file by removing system/hidden attributes first"""
    try:
        if os.path.exists(path):
            os.system(f'attrib -s -h "{path}"')
            os.remove(path)
    except Exception as e:
        pass  # Silently handle non-existent files

def safe_create_dir(path):
    """Safely create directory by removing attributes first"""
    try:
        if os.path.exists(path):
            os.system(f'attrib -r -s -h "{path}"')
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {path}: {e}")

def apply_to_target(output_dir, target_path, is_drive=False):
    """Copy files and set attributes for target path"""
    source_dir = output_dir / ('drive' if is_drive else 'folder')
    target_path = Path(target_path)
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' not found. Run script without --apply first.")
        return False
    
    print(f"\nApplying icons to: {target_path}")
    
    # Handle drive paths specially on Windows
    if is_drive and os.name == 'nt':
        # Convert G: to G:\ format
        target_path = Path(f"{str(target_path.drive)}\\")
    
    # Remove existing files first
    if is_drive:
        safe_remove(target_path / '.VolumeIcon.ico')
        safe_remove(target_path / 'autorun.inf')
    else:
        safe_remove(target_path / 'folder.ico')
        safe_remove(target_path / 'desktop.ini')
    
    try:
        # Copy files and verify
        for file in os.listdir(source_dir):
            src_file = source_dir / file
            dst_file = target_path / file
            shutil.copy2(src_file, dst_file)
            if not dst_file.exists():
                print(f"Warning: Failed to copy {file}")
            else:
                print(f"Copied: {file}")
        
        # Set attributes
        if is_drive:
            # Update autorun.inf content for drive
            autorun_content = f"""[autorun]
icon=.VolumeIcon.ico
label={target_path.drive}
action=Open folder to view files
shell\\open=Open folder to view files
shell\\open\\command=explorer.exe
UseAutoPlay=1"""
            
            try:
                with open(target_path / 'autorun.inf', 'w', encoding='utf-8') as f:
                    f.write(autorun_content)
            except Exception as e:
                print(f"Warning: Could not write autorun.inf: {e}")
                
            set_drive_attributes(target_path)
            
            print("\nDrive icon applied! Please:")
            print("1. Run: python icon_maker.py --refresh")
            print("2. Disconnect and reconnect the drive")
            print("3. If still not working, try:")
            print("   - Run script as administrator")
            print("   - Check if drive is NTFS formatted")
            print("   - Verify Group Policy allows AutoRun")
        else:
            set_folder_attributes(target_path)
        
        return True
        
    except Exception as e:
        print(f"Error applying icons: {e}")
        return False

def create_all_icons(svg_url, target_path=None, is_drive=False):
    # Create output directory
    output_dir = Path('icon_output')
    safe_create_dir(output_dir)
    
    # Create and clean subdirectories
    for subdir in ['drive', 'folder', 'mac']:
        dir_path = output_dir / subdir
        safe_create_dir(dir_path)
        
        # Clean existing files
        if subdir == 'drive':
            safe_remove(dir_path / '.VolumeIcon.ico')
            safe_remove(dir_path / 'autorun.inf')
        elif subdir == 'folder':
            safe_remove(dir_path / 'folder.ico')
            safe_remove(dir_path / 'desktop.ini')
    
    # Convert SVG to PNG in memory
    png_data = cairosvg.svg2png(url=svg_url, output_width=1024, output_height=1024)
    
    # Open the PNG data
    img = Image.open(io.BytesIO(png_data))
    
    # Create Windows ICO with 256x256 as primary size
    ico_sizes = [(256, 256)]  # Only use 256x256 for drive icon
    ico_images = []
    
    # Create high-quality resized image
    resized_img = img.resize((256, 256), Image.Resampling.LANCZOS)
    # Convert to RGBA if not already
    if resized_img.mode != 'RGBA':
        resized_img = resized_img.convert('RGBA')
    ico_images.append(resized_img)
    
    # Save drive icons with explicit sizes list
    ico_images[0].save(
        output_dir / 'drive' / '.VolumeIcon.ico',
        format='ICO',
        sizes=[(256, 256)],  # Explicit list
        optimize=False
    )

    # Update autorun.inf to explicitly specify the icon
    with open(output_dir / 'drive' / 'autorun.inf', 'w', encoding='utf-8') as f:
        f.write('''[autorun]
icon=.VolumeIcon.ico
''')

    # For folders, keep the multiple sizes
    folder_ico_sizes = [(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)]
    folder_ico_images = []
    
    # Create base image for folder icon
    base_img = img.convert('RGBA')
    
    # Create all sizes first
    for size in folder_ico_sizes:
        resized = base_img.resize(size, Image.Resampling.LANCZOS)
        folder_ico_images.append(resized)
    
    # Save folder icon with all sizes at once
    folder_ico_images[0].save(
        output_dir / 'folder' / 'folder.ico',
        format='ICO',
        append_images=folder_ico_images[1:],  # Add all other sizes
        sizes=folder_ico_sizes,  # Specify all sizes explicitly
        optimize=False
    )

    # Create desktop.ini for folders
    with open(output_dir / 'folder' / 'desktop.ini', 'w', encoding='utf-8') as f:
        f.write('''[.ShellClassInfo]
IconResource=folder.ico,0
[ViewState]
Mode=
Vid=
FolderType=Generic''')

    # Create ICNS-compatible PNGs
    icns_sizes = [
        (16, '16x16'),
        (32, '16x16@2x'),
        (32, '32x32'),
        (64, '32x32@2x'),
        (128, '128x128'),
        (256, '128x128@2x'),
        (256, '256x256'),
        (512, '256x256@2x'),
        (512, '512x512'),
        (1024, '512x512@2x')
    ]
    
    # Generate each icon size as PNG for macOS
    for size, name in icns_sizes:
        resized_img = img.resize((size, size), Image.Resampling.LANCZOS)
        resized_img.save(output_dir / 'mac' / f'icon_{name}.png')
    
    # Try to set file attributes on Windows
    if os.name == 'nt':
        try:
            # Set attributes for output directory files
            set_drive_attributes(output_dir / 'drive')
            set_folder_attributes(output_dir / 'folder')
            
            # If target path specified, apply icons there
            if target_path:
                target_path = Path(target_path)
                if target_path.exists():
                    apply_to_target(output_dir, target_path, is_drive)
                    print(f"\nIcons applied to: {target_path}")
                else:
                    print(f"\nWarning: Target path {target_path} does not exist")
            
            print("File attributes set successfully!")
        except Exception as e:
            print(f"Note: Could not set file attributes. Error: {e}")
    
    print(f"""
All icons created successfully in the 'icon_output' directory!

To apply to a new location later, run:
    python {sys.argv[0]} --apply [path] [--drive]

For macOS:
- Transfer the 'mac' folder to a Mac
- Rename it to 'fox.iconset'
- Run: iconutil -c icns fox.iconset -o .VolumeIcon.icns
    """)

if __name__ == "__main__":
    import argparse
    
    description = """
Icon Maker - Create and apply custom icons for Windows drives and folders

This script creates and applies custom icons from SVG files. It handles:
- Drive icons (.VolumeIcon.ico and autorun.inf)
- Folder icons (folder.ico and desktop.ini)
- macOS icons (iconset format)

Basic Usage:
1. Create icons:     python icon_maker.py
2. Apply to drive:   python icon_maker.py --apply G: --drive
3. Apply to folder:  python icon_maker.py --apply "path/to/folder"

Note: Some operations may require administrator privileges, especially:
- Setting drive icons
- Modifying system registry
- Refreshing icon cache
"""

    epilog = """
Examples:
  Create icons:
    python icon_maker.py
  
  Apply to drive:
    python icon_maker.py --apply G: --drive
  
  Apply to folder:
    python icon_maker.py --apply "C:/Users/username/Documents/my_folder"
  
  Force refresh icons (if changes don't appear):
    python icon_maker.py --refresh

Technical Details:
- Drive icons: Creates .VolumeIcon.ico and autorun.inf with system attributes
- Folder icons: Creates folder.ico and desktop.ini with system attributes
- Registry: Modifies Windows registry for drive icon persistence
- Icon Cache: Can clear Windows icon cache to force refresh
- macOS: Creates .iconset directory with all required sizes

Note: The --refresh command will:
1. Stop Windows Explorer
2. Clear icon cache files
3. Restart Explorer
Use with caution and save all work before running.
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--apply', 
                       metavar='PATH',
                       help='Apply icons to specified path (drive or folder)')
    
    parser.add_argument('--drive',
                       action='store_true',
                       help='Treat the target path as a drive (for --apply)')
    
    parser.add_argument('--refresh',
                       action='store_true',
                       help='Force refresh Windows icon cache (may require admin rights)')
    
    args = parser.parse_args()
    
    if args.refresh:
        refresh_windows_icons()
    elif args.apply:
        # Apply existing icons to new location
        output_dir = Path('icon_output')
        if not output_dir.exists():
            print("Error: icon_output directory not found. Run script without --apply first.")
        else:
            success = apply_to_target(output_dir, args.apply, args.drive)
            if success:
                print(f"Icons applied to: {args.apply}")
    else:
        # Create new icons
        svg_url = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f98a.svg"
        create_all_icons(svg_url)