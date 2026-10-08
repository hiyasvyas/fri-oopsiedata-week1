# Spotting the moment a robot fails on OopsieData

Annie Li, Phoebe Wang, and Hiya Vyas. FRI II Robot Learning, The University of Texas at Austin.

This repo follows `proposal/project_proposal.tex`. The goal is to name the picture where a recorded robot attempt starts to fail. A frozen R3M encoder turns each picture into 2,048 numbers. Four small heads will later share one alarm. No head has been trained. The Hugging Face dataset was not edited, and nothing was uploaded back to it.

## What is done

Claas Voelcker shared OopsieData v0.1, a private Hugging Face dataset of about 53 GB and 10,677 episodes. This laptop does not have room for the full release, so we downloaded the index first, then two whole tasks.

The success mark is not in `episodes.csv`. It is inside each episode file at `episode_annotations/<operator name>/success`. `1.0` means the attempt worked. `0.0` means it failed. The labs wrote those marks. We only read them. The file does not say which picture the failure started on.

Two tasks are on the laptop, about 370 MB together:

| Task | Episodes | Successes | Failures | What we use |
| --- | --- | --- | --- | --- |
| Araya, "Open the oven" | 200 | 52, operator `LN` | 148, operator `LN` | The table-view video, `image` |
| Harvard EML, "Stack and pack the cups." | 200 | 74, operator `HXS` | 126, operator `imported_source_label` | The base-view video, `base_image` |

The oven folder has 600 videos because each attempt was saved from three cameras. One attempt is one `.h5` file plus its main-camera `.mp4`. The 200 cup attempts and the 200 oven attempts are separate tries, not pieces of one recording.

R3M is installed from https://github.com/facebookresearch/r3m. The weights are `resnet50`, about 392 MB, stored locally in `.r3m` and not in this repo. R3M was not trained.

Two R3M runs are in this repo:

- `week1/r3m_one_frame.pt` is the first run. One picture from a 13-frame oven clip. That clip is too short to be an experiment task. The file holds one vector of 2,048 numbers.
- `week1/r3m_two_tasks.pt` is the second run, on the two real tasks. Four attempts only: one oven success, one oven failure, one cup success, one cup failure. The script counted the pictures in each video, kept 16 pictures spread from the first to the last, and saved one vector per picture. Each of the four blocks has shape 16 ├ù 2048. The other 396 attempts have not been through R3M.

The script that made the second file is `week1/run_r3m_two_tasks.py`. To look at the saved numbers:

```powershell
py -3.12 -c "import torch; x=torch.load(r'week1/r3m_two_tasks.pt', weights_only=False); print(list(x)); v=x['open_oven_failure']['embedding'][0]; print(v.shape); print(round(float(v.norm()), 2))"
```

Robometer's code was cloned from https://github.com/robometer/robometer. It did not run. This laptop has no NVIDIA GPU, and the Robometer-4B weights were not downloaded. The clone is not in this repo.

## Files in this repo

| File | What it is |
| --- | --- |
| `proposal/project_proposal.tex` | The FRI proposal |
| `week1/week1_label_note.txt` | The first two episode files we opened |
| `week1/task_counts.txt` | Episode counts by folder. "fail 479" there is a path-name count, not a count of failed attempts |
| `week1/r3m_result.txt` | The one-frame R3M run, in words |
| `week1/r3m_one_frame.pt` | One 2,048-number vector from the short clip |
| `week1/run_r3m_two_tasks.py` | How the four real attempts were read and embedded |
| `week1/r3m_two_tasks.pt` | Four attempts, 16 vectors each |
| `week1/week1_robometer_note.txt` | Why Robometer did not run here |
| `download_data.py` | Downloads the WebVisualizer dataset into `data/` after `hf auth login`. That folder is gitignored |

Videos, episode files, `episodes.csv`, and the full manifest stay on the local copy at `C:\Users\sunee\oopsiedata-v0.1`. They are listed in `.gitignore`.

## Setup for download_data.py

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U huggingface_hub datasets
hf auth login
python download_data.py
```

## What is next

Watch about 15 oven attempts and about 15 cup attempts, main camera only. Check that a success and a failure start on the same kind of scene and diverge where the attempt goes wrong. Spot-check cup failures, because those marks came from `imported_source_label` and the successes came from operator `HXS`. Then write the loader that returns 16 evenly spaced frames and the existing success mark for every episode.

The full R3M cache, Robometer inference, and the four heads belong on TACC. We still need to learn how to use that machine and when to move this work off the laptop. Failure onset times are not in the files we opened. If we need timing, two people will later mark about 40 failed videos, with 10 marked by both.

Week 6 is the report only. If we fall behind, we drop cross-task evaluation, then the hand-marked onset times, then Head C. Head A against Head B is the smallest comparison that can still reject the claim.
