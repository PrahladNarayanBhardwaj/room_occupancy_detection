# Room Occupancy Estimation

Can a room's environmental sensors (temperature, light, sound, CO₂ and motion) tell whether a meeting room is in use, reliably enough to release "ghost" bookings, without cameras or tracking anyone?

I built this project on the public UCI Room Occupancy Estimation dataset (CC BY 4.0). I started with a straightforward analysis (`occupancy_test.ipynb`), then went back and stress-tested my own work much harder (`analysis/`), and wrote it up as a paper (`paper/`).

## What's in here

| Path | What it is |
|---|---|
| `occupancy_test.ipynb` | My first analysis |
| `occupancy_decision_packet.json` | The summary of my first analysis |
| `analysis/occupancy_reanalysis.ipynb` | My re-analysis. It runs top to bottom in about 3 minutes and produces every number and figure in the paper |
| `analysis/build_notebook.py` | The script I use to generate the re-analysis notebook |
| `analysis/reanalysis_results.json` | All re-analysis numbers in one file |
| `paper/Room_Occupancy_Paper.pdf` | The paper |
| `paper/build_paper.js` | Builds the Word version of the paper from the results file (`npm install docx`, then `node paper/build_paper.js`) |
| `paper/figures/` | The figures, written by the notebook |

## What I found

- **Light made my model worse.** In the training days the lights were on whenever people were in, so my model learned "lights off = empty". It missed 184 of 294 occupied snapshots when three people worked with the lights off. A random forest fails in exactly the same way, so the problem is in the data, not the algorithm.
- **My single test split hid a problem.** When I held out each day in turn, my final model raised 345 false alarms on an empty Christmas Day. The door motion sensor fired 7 times with nobody there.
- **Requiring corroborating evidence fixes it.** Counting motion firings and pairing them with CO₂ and sound gave my best model: LightGBM with F1 0.974 and a 0.9% false-alarm rate. It also holds up when I simulate a person sitting still.
- **My "10-minute" release rule really frees rooms 12–28 minutes after people leave.**
- **Random splits are misleading for this data.** Head-count accuracy is 99.8% with a random split but 86% when I hold out whole days.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
cd analysis && jupyter execute --inplace occupancy_reanalysis.ipynb
```

## Use of AI tools

I used Claude (Anthropic), an AI assistant, to learn and help write code wherever I needed it, and to help with grammar, wording and drafting of the text. I reviewed all code, results and text, and take full responsibility for everything here.
