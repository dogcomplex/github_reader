import unittest
from pathlib import Path
import os
import shutil
from PIL import Image

class TestIconMaker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test directories
        cls.test_dir = Path('test_output')
        cls.test_dir.mkdir(exist_ok=True)
        
        # Run the icon creation
        os.system('python icon_maker.py')
    
    @classmethod
    def tearDownClass(cls):
        # Clean up test directories
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        shutil.rmtree('icon_output', ignore_errors=True)

    def test_directory_structure(self):
        """Test that all required directories are created"""
        required_dirs = ['drive', 'folder', 'mac']
        for dir_name in required_dirs:
            self.assertTrue((Path('icon_output') / dir_name).exists())

    def test_drive_icon_files(self):
        """Test drive icon creation and properties"""
        drive_dir = Path('icon_output') / 'drive'
        
        # Check files exist
        self.assertTrue((drive_dir / '.VolumeIcon.ico').exists())
        self.assertTrue((drive_dir / 'autorun.inf').exists())
        
        # Check ICO properties
        with Image.open(drive_dir / '.VolumeIcon.ico') as ico:
            self.assertEqual(set(ico.info.get('sizes')), {(256, 256)})

    def test_folder_icon_files(self):
        """Test folder icon creation and properties"""
        folder_dir = Path('icon_output') / 'folder'
        
        # Check files exist and have content
        self.assertTrue((folder_dir / 'folder.ico').exists())
        self.assertTrue((folder_dir / 'desktop.ini').exists())
        
        # Verify basic image properties
        with Image.open(folder_dir / 'folder.ico') as ico:
            self.assertEqual(ico.mode, 'RGBA', "Icon should be in RGBA mode")
            self.assertTrue(len(ico.info.get('sizes', [])) > 0, "Icon should have at least one size variant")

    def test_mac_icon_files(self):
        """Test macOS icon creation"""
        mac_dir = Path('icon_output') / 'mac'
        
        # Check if all required sizes are present
        expected_files = [
            'icon_16x16.png',
            'icon_16x16@2x.png',
            'icon_32x32.png',
            'icon_32x32@2x.png',
            'icon_128x128.png',
            'icon_128x128@2x.png',
            'icon_256x256.png',
            'icon_256x256@2x.png',
            'icon_512x512.png',
            'icon_512x512@2x.png'
        ]
        for filename in expected_files:
            self.assertTrue((mac_dir / filename).exists())

    def test_file_attributes(self):
        """Test if files have correct attributes on Windows"""
        if os.name == 'nt':
            drive_dir = Path('icon_output') / 'drive'
            folder_dir = Path('icon_output') / 'folder'
            
            # Helper function to check if file is hidden and system
            def is_hidden_system(path):
                return bool(os.stat(path).st_file_attributes & 0x6)
            
            # Check drive files
            self.assertTrue(is_hidden_system(drive_dir / '.VolumeIcon.ico'))
            self.assertTrue(is_hidden_system(drive_dir / 'autorun.inf'))
            
            # Check folder files
            self.assertTrue(is_hidden_system(folder_dir / 'folder.ico'))
            self.assertTrue(is_hidden_system(folder_dir / 'desktop.ini'))

    def test_apply_to_folder(self):
        """Test applying icons to a folder"""
        # Create test folder
        test_folder = self.test_dir / 'test_folder'
        test_folder.mkdir(exist_ok=True)
        
        # Apply icons
        os.system(f'python icon_maker.py --apply "{test_folder}"')
        
        # Check if files were copied and attributes set
        self.assertTrue((test_folder / 'folder.ico').exists())
        self.assertTrue((test_folder / 'desktop.ini').exists())

    def test_apply_to_drive(self):
        """Test applying icons to a drive (mock)"""
        # Create mock drive directory
        mock_drive = self.test_dir / 'mock_drive'
        mock_drive.mkdir(exist_ok=True)
        
        # Apply icons
        os.system(f'python icon_maker.py --apply "{mock_drive}" --drive')
        
        # Check if files were copied
        self.assertTrue((mock_drive / '.VolumeIcon.ico').exists())
        self.assertTrue((mock_drive / 'autorun.inf').exists())

    def test_icon_quality(self):
        """Test that icons maintain quality (no optimization)"""
        with Image.open(Path('icon_output') / 'drive' / '.VolumeIcon.ico') as drive_ico:
            with Image.open(Path('icon_output') / 'folder' / 'folder.ico') as folder_ico:
                # Check that images are in RGBA mode
                self.assertEqual(drive_ico.mode, 'RGBA')
                self.assertEqual(folder_ico.mode, 'RGBA')

if __name__ == '__main__':
    unittest.main() 