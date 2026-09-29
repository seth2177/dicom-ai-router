"""One command, whole pipeline. The demo lives in airouter/demo.py; installed, it is `dicom-ai-router demo`.

  python run_demo.py                 # 12 studies
  python run_demo.py --count 20 --seed 3
  python run_demo.py --fail-rate 0.4 # watch the router retry a flaky AI endpoint
"""
from airouter.demo import main

if __name__ == "__main__":
    main()
