import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from obsidian_sync import ObsidianSync
ObsidianSync().push_all()