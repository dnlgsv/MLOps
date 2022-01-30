from pathlib import Path

def test_dataset_existence():
  assert Path("/path/to/file").is_file(), "There is no dataset 'wine_quality.csv' to apply model on."
