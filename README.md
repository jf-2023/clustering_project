# SEC Data Key Equivalence Finder

## Overview

This script aims to identify equivalent data keys across different entities in the SEC (U.S. Securities and Exchange Commission) API. Each entity may use different keys to represent similar data points (e.g., "REVENUES" for AAPL vs. "SALES" for GOOG). The goal is to cluster these keys based on semantic similarity to enhance data retrieval completeness.

## Requirements

- Python 3.6 or above
- Libraries:
  - `sentence_transformers`
  - `scikit-learn`
  - `json`
  - `cProfile`
  - `pstats`
  - `tqdm`

## Installation

1. Clone the repository:

2. Install dependencies:
   
    ```bash
   pip install -r requirements.txt



## New Features