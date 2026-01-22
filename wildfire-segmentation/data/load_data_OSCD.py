import os
import pathlib

from datasets import load_dataset
oscd_dataset = load_dataset("blanchon/OSCD_MSI")
print(oscd_dataset)