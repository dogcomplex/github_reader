from PIL import Image
import os
from pathlib import Path
import cairosvg
import io
import shutil
import sys
import ctypes
import requests
import json
from functools import lru_cache

def is_admin():
    """Check if the script is running with admin rights"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def elevate_if_needed():
    """Re-run the script with admin rights if needed"""
    if not is_admin():
        print("Attempting to elevate privileges...")
        try:
            args = ' '.join(sys.argv)
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, args, None, 1
            )
            sys.exit(0)
        except Exception as e:
            print(f"Could not elevate privileges: {e}")
            return False
    return True

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
    try:
        # Set attributes for the icon and desktop.ini
        os.system(f'attrib +s +h "{folder_path}/desktop.ini"')
        os.system(f'attrib +s +h "{folder_path}/folder.ico"')
        
        # Alternative 1: Use System folder type
        desktop_ini_content = """[.ShellClassInfo]
IconResource=folder.ico,0
IconFile=folder.ico
IconIndex=0
[ViewState]
Mode=
Vid=
FolderType=System"""

        # Alternative 2: Add more shell class registrations
        desktop_ini_content_alt = """[.ShellClassInfo]
IconResource=folder.ico,0
IconFile=folder.ico
IconIndex=0
[{BE098140-A513-11D0-A3A4-00C04FD706EC}]
IconArea_Image=folder.ico
[ViewState]
Mode=
Vid=
FolderType=Generic"""

        # Write the enhanced desktop.ini
        with open(folder_path / 'desktop.ini', 'w', encoding='utf-8') as f:
            f.write(desktop_ini_content)  # or desktop_ini_content_alt
            
    except Exception as e:
        print(f"Warning: Could not set all folder attributes: {e}")

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
    """Apply icons to target location"""
    try:
        target_path = Path(target_path)
        
        # Check if we need admin rights
        needs_admin = False
        try:
            test_file = target_path / '.test_write'
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            needs_admin = True
        
        if needs_admin and not is_admin():
            print("\nThis operation requires administrator privileges.")
            print("Please run the script as administrator or use the --force option to attempt elevation.")
            return False
            
        # Remove existing files first
        if is_drive:
            safe_remove(target_path / '.VolumeIcon.ico')
            safe_remove(target_path / 'autorun.inf')
        else:
            safe_remove(target_path / 'folder.ico')
            safe_remove(target_path / 'desktop.ini')
        
        # Copy new files
        if is_drive:
            shutil.copy2(output_dir / 'drive' / '.VolumeIcon.ico', target_path)
            shutil.copy2(output_dir / 'drive' / 'autorun.inf', target_path)
            set_drive_attributes(target_path)
        else:
            # Copy icon file first
            shutil.copy2(output_dir / 'folder' / 'folder.ico', target_path)
            
            # Create desktop.ini with correct path
            desktop_ini_content = """[.ShellClassInfo]
IconResource=folder.ico,0
IconFile=folder.ico
IconIndex=0
[ViewState]
Mode=
Vid=
FolderType=Generic"""
            
            # Write desktop.ini directly to target
            with open(target_path / 'desktop.ini', 'w', encoding='utf-8') as f:
                f.write(desktop_ini_content)
            
            # Set attributes in correct order
            os.system(f'attrib +s +h "{target_path}/folder.ico"')
            os.system(f'attrib +s +h "{target_path}/desktop.ini"')
            os.system(f'attrib +r "{target_path}"')
            
        return True
        
    except Exception as e:
        print(f"Error applying icons: {e}")
        return False
        
    # Rest of the function remains the same...

def convert_image_to_png(image_path):
    """Convert any supported image to PNG format in memory"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create in-memory PNG
            png_buffer = io.BytesIO()
            img.save(png_buffer, format='PNG')
            return png_buffer.getvalue()
    except Exception as e:
        raise ValueError(f"Could not process image: {e}")

def create_all_icons(source, target_path=None, is_drive=False):
    """Modified to handle both URLs and local files"""
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
    
    # Handle source based on type
    if isinstance(source, str) and (source.startswith('http://') or source.startswith('https://')):
        # Existing SVG URL handling
        png_data = cairosvg.svg2png(url=source, output_width=1024, output_height=1024)
    else:
        # Local file handling
        png_data = convert_image_to_png(source)
    
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

def fetch_emoji_metadata():
    """Fetch and parse emoji metadata from Unicode and Twemoji"""
    try:
        # Fetch from Unicode.org
        unicode_url = "https://unicode.org/Public/emoji/latest/emoji-test.txt"
        response = requests.get(unicode_url, timeout=5)
        response.raise_for_status()
        
        # Parse Unicode data for names and codes
        emoji_data = {}
        for line in response.text.split('\n'):
            if '; fully-qualified' in line:
                code = line.split(';')[0].strip().lower().replace(' ', '')
                name = line.split('#')[1].split(' ', 2)[2].strip().lower().split()[0]
                emoji_data[name] = code  # Store name -> code mapping
                emoji_data[code] = code  # Also store direct code -> code mapping
        
        return emoji_data
            
    except Exception as e:
        print(f"\nWarning: Could not fetch remote emoji list: {e}")
        print("Falling back to basic emoji set...\n")
        fallback = {
            'fox': '1f98a',
            'lotus': '1fab7',
            'paw_prints': '1f43e',
            'dragon': '1f409',
            'wolf': '1f43a',
            'cat': '1f431',
            'dog': '1f436',
            'unicorn': '1f984',
            'phoenix': '1f985',
            'butterfly': '1f98b',
            'rose': '1f339',
        }
        # Add direct code mappings to fallback
        return {**fallback, **{v: v for v in fallback.values()}}

def list_available_emojis():
    """Display available emojis in a formatted table"""
    emojis = fetch_emoji_metadata()
    
    print("\nAvailable Emojis:")
    print("=" * 50)
    print(f"{'Name/Code':<15} {'Code':<10} {'Symbol':<10}")
    print("-" * 50)
    
    # Only show name->code mappings (skip code->code duplicates)
    shown_codes = set()
    sorted_emojis = sorted((k, v) for k, v in emojis.items() if not k.startswith('1f'))
    
    for name, code in sorted_emojis:
        if code in shown_codes:
            continue
        shown_codes.add(code)
        try:
            symbol = chr(int(code, 16))
            print(f"{name:<15} U+{code.upper():<10} {symbol:<10}")
        except ValueError:
            print(f"{name:<15} U+{code.upper():<10} {'?':<10}")
    
    print("\nUsage:")
    print("  python icon_maker.py --emoji <name>     # Use friendly name (e.g., 'fox')")
    print("  python icon_maker.py --emoji <code>     # Use Unicode code (e.g., '1f98a', '1f339')")
    print("\nNote: Any valid Unicode emoji code will work, even if not listed above.")
    return emojis

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
    
    parser.add_argument('--force', 
                       action='store_true',
                       help='Attempt to elevate privileges if needed')
    
    parser.add_argument('--list',
                       action='store_true',
                       help='List all available emoji options')
    
    parser.add_argument('--image',
                       metavar='PATH',
                       help='Use local image file (PNG, JPEG, WEBP, etc.) as icon source')
    
    # Parse initial arguments
    args, remaining = parser.parse_known_args()
    
    if args.list:
        list_available_emojis()
        sys.exit(0)
    
    # Get emoji list for --emoji choices
    emojis = fetch_emoji_metadata() or {
        'fox': '1f98a',
        'lotus': '1fab7',
        'paw_prints': '1f43e',
        'dragon': '1f409',
        'wolf': '1f43a',
        'cat': '1f431',
        'dog': '1f436',
        'unicorn': '1f984',
        'phoenix': '1f985',
        'butterfly': '1f98b',
        'rose': '1f339',
    }
    
    # Add emoji argument after getting the list
    parser.add_argument('--emoji',
                       help='''Choose emoji to use as icon. Can be:
                           - A friendly name (e.g., 'fox', 'lotus')
                           - A Unicode code (e.g., '1f98a', '1f339')
                           Any valid Unicode emoji code will work.
                           Use --list to see available friendly names.''')
    
    # Parse all arguments again
    args = parser.parse_args()
    
    # Handle elevation if needed
    if args.force and any([args.apply, args.refresh]):
        if not elevate_if_needed():
            print("Could not get required permissions. Try running as administrator.")
            sys.exit(1)
    
    if args.refresh:
        refresh_windows_icons()
    elif args.apply:
        output_dir = Path('icon_output')
        if not output_dir.exists():
            print("Error: icon_output directory not found. Run script without --apply first.")
        else:
            success = apply_to_target(output_dir, args.apply, args.drive)
            if success:
                print(f"Icons applied to: {args.apply}")
            else:
                print("\nTo retry with elevated privileges, use:")
                print(f"python {sys.argv[0]} --apply \"{args.apply}\" {'--drive' if args.drive else ''} --force")
    else:
        if args.image:
            create_all_icons(args.image)
        elif args.emoji:
            # Existing emoji handling
            emoji_code = args.emoji.lower()
            if emoji_code not in emojis:
                emoji_code = emoji_code.replace('u+', '').replace('0x', '')
            else:
                emoji_code = emojis[emoji_code]
            
            svg_url = f"https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/{emoji_code}.svg"
            create_all_icons(svg_url)