# Taiyosha TCapture Backend

This repository contains the backend for Taiyosha TCapture.

## Prerequisites

- Python 3.9 installed
- Recommended: Virtual environment for dependency management (Conda is recommended)

## Setup & Installation

1. Clone the repository and navigate to the backend directory:

    ```sh
    git clone <repository_url>
    ```

2. Create and activate a virtual environment:

    ```sh
    conda create --name tcapture python=3.9 -y
    ```

3. Install dependencies:

    ```sh
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu     #cpu
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118   #cuda-11.8
    pip install -r requirements.txt
    ```

## Running the Application

Start the backend server by executing:

```sh
python main.py
