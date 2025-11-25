import sys
import os
from pathlib import Path
from typing import ClassVar, Final, Optional, Union

class NotebookInitializer:
    """
    Initializes the execution environment for Jupyter notebooks within a project.

    This utility ensures the working directory is set to the project root and 
    provides standardized access paths for data and configuration.
    """

    # Class-level constants for directory structure
    _ROOT_TOKEN: Final[str] = 'notebooks'
    
    # Instance attributes
    root_dir: Optional[Path] = None
    src_dir: Optional[Path] = None
    data_dir: Final[Path]
    raw_data_dir: Final[Path]
    processed_data_dir: Final[Path]
    
    # Placeholder for configuration manager instance
    config: Final
    
    @classmethod
    def get_instance(cls, script_path: Union[str, Path]):
        if cls.root_dir is None:
            cls.root_dir = Path(script_path).resolve().parent
        return cls

    def __init__(self, script_path: Union[str, Path]) -> None:
        """
        Calculates project paths based on the location of the initializing script.

        Args:
            script_path (Union[str, Path]): The path to the currently executing script
                (usually Path(__file__).resolve()).
        """
        # Calculate ROOT_DIR by moving two levels up from the script's path 
        # (assuming the script is inside 'notebooks/' or similar).
        self.root_dir = NotebookInitializer.get_instance(script_path).root_dir
        if self.root_dir is None:
            print(f"ROOT_DIR not set. Setting to: {Path(script_path).resolve().parent}")
            self.root_dir = Path(script_path).resolve().parent
        else:
            print(f"ROOT_DIR already set to: {self.root_dir}")

        # Define directory paths
        self.data_dir = self.root_dir / "data"
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.models_dir = self.root_dir / "models"


    def setup_environment(self) -> None:
        """
        Changes the current working directory to the project root and reports the setup.
        """
        print(f"Original working directory: {os.getcwd()}")
        
        # Change the current working directory to the project root
        os.chdir(self.root_dir)
        
        print(f"Current working directory changed to the project root: {os.getcwd()}")
        
        # Initialize configuration manager
        from config.manager import ConfigManager
        self.config = ConfigManager()
        
        # Verify and report the directory structure
        self._report_structure()

    def _report_structure(self) -> None:
        """Prints the standardized directory structure for verification."""
        print("\n--- Directory Structure Setup ---")
        print(f"📂 Root Directory: {self.root_dir}")
        print(f"📁 Data Directory: {self.data_dir}")
        print(f"📥 Raw Data Directory: {self.raw_data_dir}")
        print(f"📤 Processed Data Directory: {self.processed_data_dir}")
        print(f"⚙️ Config Manager: {self.config}")
        print(f"⚙️ Models Directory: {self.models_dir}")
