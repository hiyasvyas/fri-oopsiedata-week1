# Week 1 progress: spotting the moment a robot fails on OopsieData

Annie Li, Phoebe Wang, and Hiya Vyas. FRI II Robot Learning, The University of Texas at Austin.

This repo is the week 1 record for the project in `proposal/project_proposal.tex`. Week 1 in that proposal is: get the OopsieData episodes, check whether each episode is already marked success or failure, and get R3M and Robometer to run once. No head was trained. The Hugging Face dataset was not edited and nothing was uploaded back to it.

## What week 1 finished

Claas Voelcker shared OopsieData v0.1, a private Hugging Face dataset of about 53 GB. We downloaded the index and three small episode files onto this computer. The index, `episodes.csv`, lists 10,677 episodes. That spreadsheet does not say success or failure.

The mark is inside each episode file. We opened two files and did not change them:

- `labs/Araya/flower_open_oven/20260803_211207.h5`. The operator group is `LN`. `success` is 1.0. The taxonomy says the outcome is success. The instruction is "Open the oven".
- `labs/Berkeley_RAIL/20260614_125000/20260614_125502.h5`. The operator group is `jagdeep`. `success` is 1.0. The instruction is "Move the left stuffed toy to the largest bowl."

The field is `episode_annotations/<operator name>/success`. The operator name changes from file to file. Neither file has the frame where a failure starts. Videos are separate `.mp4` files named inside the episode file.

R3M, the frozen picture network from the proposal, ran on the first frame of one of those videos. The frame is 224 by 224. The description is 2,048 numbers. That vector is `week1/r3m_one_frame.pt`. R3M was not trained.

Robometer's public code was cloned from https://github.com/robometer/robometer. The inference script is `scripts/example_inference_local.py`. The published model is Robometer-4B. It did not run here, because this laptop has no NVIDIA GPU. The proposal runs that model on TACC. The weights were not downloaded, and nothing was trained. The clone is not in this repo, because it is someone else's code and we did not change it.

## Files in this repo

| File | What it is |
| --- | --- |
| `proposal/project_proposal.tex` | The FRI proposal this week follows |
| `week1/week1_label_note.txt` | What the two episode files contain |
| `week1/task_counts.txt` | How many episodes sit in each task folder with at least 20 episodes |
| `week1/r3m_result.txt` | The one R3M run, in words |
| `week1/r3m_one_frame.pt` | The 2,048-number description from that frame |
| `week1/week1_robometer_note.txt` | Why Robometer did not run on this laptop |

`task_counts.txt` ends with word counts inside file paths. "fail 479" means the letters f-a-i-l appear in 479 paths. It is not a count of failed attempts. The real success mark is inside the episode files.

## What was looked at, and what was not changed

Looked at, locally:

- `C:\Users\sunee\oopsiedata-v0.1\episodes.csv`
- `C:\Users\sunee\oopsiedata-v0.1\README.md` from the dataset
- the two `.h5` files named above
- `labs/Araya/flower_open_oven/20260803_211207_image.mp4`, first frame only

Not included here, on purpose: the videos, the episode files, and the full index. Those stay on the local copy. A third small file, `labs/Araya/dit_close_oven/20260803_202945.h5`, did not download.

## What week 2 is

Watch about 30 attempts, pick two tasks that differ, and check that a success and a failure of the same instruction share the scene until the mistake. The proposal's examples are placement and pouring. This release has cup tasks, including `Harvard_EML/stack-and-pack-cups` (200 episodes) and `Harvard_EML/pack-cup` (99). Path names contain the word "pour" 18 times and "bin" 2 times, so those exact examples may be rare and the two tasks should be chosen after watching.

## Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -U huggingface_hub datasets
hf auth login                   # paste YOUR OWN Hugging Face token
python download_data.py
```

Data downloads into `data/` (gitignored).
