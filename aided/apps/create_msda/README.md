# Create msda

Example README.md to populate for sphinx on create msda app

## Usage

```
usage: aided create-msda [-h] -l LOG_FILE -o OUTPUT_FILE [-T T]

options:
  -h, --help            show this help message and exit
  -l LOG_FILE, --log-file LOG_FILE
                        WFN log file
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Output MSDA file
  -T T                  Temperature in Kelvin (default: 300.0)
```

## Description

The create-msda app writes out a Mean Square Displace Amplitude matrix to a text file. It takes in a log file
from an application like Gaussian09 which performs geometry optimization and outputs vibrational frequencies.
