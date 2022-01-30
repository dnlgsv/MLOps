from pathlib import Path

def test_dataset_existence():
  assert Path("/data_governance_hw/wine_quality.csv").is_file(), "There is no dataset 'wine_quality.csv' to apply model on."
