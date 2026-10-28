"""Re-run shared independent operator tests using NGC5548 physical parameters."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
sys.path.insert(0,str(ROOT))
from run_reproduction import PARAMS
import validate_model

if __name__=="__main__":
    validate_model.ROOT=ROOT
    validate_model.model.DEFAULT.update(PARAMS)
    validate_model.main()
