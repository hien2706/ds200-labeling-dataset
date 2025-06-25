# Setup

## First: Clone the Repository

```bash
git clone https://github.com/hien2706/ds200-labeling-dataset.git
cd ds200-labeling-dataset
```

## Second: Enable the Script to Create Directory Structure

```bash
chmod +x create_directory.sh
./create_directory.sh
```

## Third: Create Virtual Environment

```bash
python -m venv ds200
source ds200/bin/activate
pip install openai
```

## Fourth: Fix CPU Count Configuration

Open the `agentA_deepseek.py` at line 433 to fix the CPU count

Do the same for `agentB_openai.py` at line 57

## Fifth: Create Environment File

Create `.env` file and add your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

# Run the Script to Label the Dataset

## For Agent A

```bash
python agentA_deepseek.py tokenized_data_5500.json
```

Tri will run from `tokenized_data_5500.json` to `tokenized_data_11000.json`

After running there will be errors. Check in `checkpoints/agentA/` and the log for that batch to find out the failed IDs. Check `logs/agentA` for failed output, then manually fix them in the result file.

After that, run `sort_and_copy.py` with the correct file_path to put into final result.

## For Agent B

```bash
python agentB_openai.py tokenized_data_5500.json
```

There will be errors. Fix them the same way as Agent A.

After that, run `sort_and_copy.py` with the correct file_path to put into final result.

[1] https://github.com/hien2706/ds200-labeling-datas