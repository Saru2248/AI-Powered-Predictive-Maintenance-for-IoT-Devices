Place your NASA CMAPSS dataset files here:

  train_FD001.txt   — Training data (engine sensor readings)
  test_FD001.txt    — Test data
  RUL_FD001.txt     — Remaining Useful Life ground truth

Also available (optional, if you downloaded all 4 sub-datasets):
  train_FD002.txt / test_FD002.txt / RUL_FD002.txt
  train_FD003.txt / test_FD003.txt / RUL_FD003.txt
  train_FD004.txt / test_FD004.txt / RUL_FD004.txt

Download source:
  https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
  OR
  https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

Column layout (no header in file):
  Col 1:  unit_id       (engine number)
  Col 2:  time_cycles   (operational cycle)
  Col 3-4: op_setting_1, op_setting_2, op_setting_3
  Col 5-26: sensor_1 ... sensor_21
