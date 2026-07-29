"""让 tests/ 里能直接 import sop 和 run。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
