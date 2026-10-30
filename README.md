# Sensors: noise or a lie?

When a group of sensors starts reporting strange values, there are two very different explanations:

- **Noise.** Every sensor gets a little shaky at once, like static on a radio. Annoying, but nobody is wrong on purpose.
- **A lie.** One or two sensors report wrong values while the rest are fine. The sensor might be broken, stuck, or tampered with.

This repository builds a monitor that first notices *that* something is wrong, then guesses *which* of the two it is. It tests that monitor on five real datasets: a gas-sensor array, an industrial oven, an elderly person's home, an office room with a head count, and people walking with and without a leg brace.

Every monitor here learns only from clean data. Nothing is tuned by hand, and every reported number comes from data the monitor never saw while it was being set up.

---

## Contents

1. [The idea in one picture](#1-the-idea-in-one-picture)
2. [How the monitor works](#2-how-the-monitor-works)
3. [How we test it](#3-how-we-test-it)
4. [Results at a glance](#4-results-at-a-glance)
5. [The five datasets](#5-the-five-datasets)
6. [What we learned](#6-what-we-learned)
7. [Which setup should I use?](#7-which-setup-should-i-use)
8. [Getting the data](#8-getting-the-data)
9. [Running it](#9-running-it)
10. [What is in the repository](#10-what-is-in-the-repository)
11. [History of changes](#11-history-of-changes)

---

## 1. The idea in one picture

Think of the sensors as **witnesses to the same event**.

- If **one witness** tells a story that does not match everyone else's, that witness is probably lying.
- If **everyone's story** is a little off, the lighting was probably bad: that is noise.

So the monitor asks one question: **how many witnesses do I have to ignore before everyone else agrees?**

| Witnesses ignored | Verdict |
|---|---|
| 0 | Nothing is wrong (no alarm) |
| 1 or 2 | **A lie**, and the ignored ones are the suspects |
| Most of them | **Noise** |

This only works when the sensors normally agree with each other. That is the single most important fact in this whole project, and it explains most of the results below.

---

## 2. How the monitor works

### Step 1: is anything wrong?

The monitor learns what "normal" looks like from clean data: the average reading and how readings normally move *together*. For a new reading it computes one number, the **Mahalanobis distance**. Think of it as "how surprised should I be?", where breaking the usual relationships between sensors counts just as much as an odd value on its own. If the distance is above a line, the alarm goes off.

### Step 2: noise or lie?

Two ways to count how many sensors are involved:

| Name | What it counts | Strength | Weakness |
|---|---|---|---|
| `hot` (the original) | Sensors whose own readings look odd *on their own* | Simple | Blind to a sensor that shows a real-looking value from another moment, because on its own it looks fine |
| `removal` (new) | Sensors that must be **set aside** before the rest agree (the witness idea) | Sees sensors that disagree with each other, and **names the suspects** | Needs sensors that normally move together |

Then a **cutoff** decides: a count below the cutoff is a lie, a count at or above it is noise. The cutoff is chosen automatically (see [section 3](#3-how-we-test-it)).

`removal` does not need any refitting. A standard identity splits the total distance into "what the other sensors explain" plus "what sensor *k* adds on its own" (`d = d_without_k + w_kᵀ (P_kk)⁻¹ w_k`, with `P` the inverse covariance and `w = P (x − μ)`). That makes it cheap to try setting sensors aside one at a time, most suspicious first.

### For sensors read over time: watch the surprise, not the value

In four of the five datasets each sensor is read repeatedly over time (every minute, every 20 seconds, ...). A value on its own says little: the oven is hot during the day and cool at night. What matters is **how the reading moved since last time**, compared with how it usually moves. Two ways to define "usual":

| Name | Expected move | Good when |
|---|---|---|
| `jump` | The average move | Readings move randomly |
| `trend` | A prediction from the sensor's own last 3 moves | Movements persist, e.g. a rising temperature keeps rising |

The **surprise** is the actual move minus the expected move. The monitor watches the last *W* surprises of each sensor (*W* = 1, or a longer window such as 10 minutes) and applies the same Step 1 and Step 2 to them.

### Where the alarm lines come from

| Name | How the line is set | Note |
|---|---|---|
| `chi2` | A textbook formula that assumes bell-curve data | Real data has fatter tails, so it usually rings far too often: about 8–11% of clean readings on gas and oven, and up to 58% on drifted gas data, instead of the intended 0.1% |
| `empirical` | Learned from clean data so that about **1%** of clean readings cross it | Honest false-alarm rate. **Use this one.** |

---

## 3. How we test it

We take clean, real data and **break it on purpose** in known ways, then check whether the monitor notices and names the right kind of problem.

| Corruption | What we do | Real-world equivalent |
|---|---|---|
| **noise** | Add random jitter to every sensor, sized as 0.1× to 4× the usual movement | Electrical interference, a bad power supply |
| **obvious lie** | 1–6 sensors show a reading copied from another, unrelated moment | A sensor swapped, mis-wired or spoofed crudely |
| **plausible lie** | 1–6 sensors replay a normal-looking stretch of movement from another time | A careful spoof, a replay attack |
| **stuck** | 1–6 sensors freeze at their last value | A frozen or disconnected sensor |
| **whole-sensor lie** (gas only) | All 8 numbers of 1–6 sensors copied from another clean reading | One sensor lying with real-looking values |
| **feature lie** (gas only) | 5–25% of single numbers scattered across many sensors copied from another reading | Scattered corruption |

**What "correct" means.** A noisy reading is correct if it is called noise. A lie is correct if it raises an alarm *and* is called a lie with at least one sensor counted. Every number in the tables is the **share of corrupted readings that were called the right kind**.

**Fair scoring.** Readings that already raised an alarm before we corrupted them are left out. So are corruptions that **did not change the reading at all**. This matters more than you would think: in the oven, readings are whole degrees and the temperature usually does not move from one minute to the next, so 61% of "plausible lies" on one thermometer left the value exactly as it was. No method could ever see those. Noise is also rounded to each sensor's own resolution (whole degrees, 0.06 °C, ...), because a real sensor can only output values on its own grid.

**No peeking.** Data is split into three parts:

1. **Training** (usually the first 70% of the time): learns what normal looks like.
2. **Calibration** (half of the rest): sets the learned lines and chooses the noise-vs-lie cutoff.
3. **Evaluation** (the other half): produces every number you see.

This split is repeated with **five different shuffles** (seeds 42–46). Tables show the median; the scripts also print the min and max. When those are close, the result is stable.

---

## 4. Results at a glance

Learned (`empirical`) lines at about 1% false alarms, median of five seeds. "Best setup" is the one that does best overall on that dataset; see each section for trade-offs.

| Dataset | Sensors agree with each other? | Best setup | Noise caught | 1-sensor obvious lie | 1-sensor plausible lie | 1 sensor stuck |
|---|---|---|---|---|---|---|
| Gas array | Yes | `removal` | 0.94 at 0.25× | 0.47 (whole-sensor lie) | – | – |
| Oven | **No** | `trend` + `removal` | 0.98 at 2× | 0.95 | 0.06 | 0.04 |
| Home | Partly (gas sensors) | `jump` + `removal` | 0.96 at 4× | 0.93 | 0.02 | 0.00 |
| Room | Partly (lights, sound) | `trend` + `removal` | 0.93 at 2× | 0.49 | 0.01 | 0.00 |
| Walking | **Yes** (left/right twins) | `trend` + `removal` | 0.96 at 1× | 0.99 | 0.95 | 0.95 |

How to read it:

- **Noise is the easy part.** With `removal`, everything noticeable gets caught.
- **Obvious lies are usually caught**, except in the room, which is empty and dark most of the time, so a copied value often looks like the real one.
- **Plausible lies and stuck sensors are only caught when the sensors are smooth or strongly coupled** (walking). Everywhere else they are mostly invisible. That is a limit of the data, not of the method: see [section 6](#6-what-we-learned).

---

## 5. The five datasets

### 5.1 Gas sensor array (`run_workshop_repeats.py`, `run_neighbor_check.py`)

**What it is.** 16 chemical sensors, 8 numbers each (128 per reading), 13,910 readings in 10 batches recorded over three years (the UCI Gas Sensor Array Drift data). Each reading is a snapshot, so there is no time series here and the monitor works on the values themselves.

**The problem we found.** A lying sensor copies a *real* reading from another moment, so its 8 numbers look perfectly normal on their own. The original `hot` count found **zero** odd sensors and threw the alarm away. Whole-sensor lies were identified correctly 0% of the time with learned lines (2–4% with `chi2`).

**The fix and the result.** Random 70/30 split, learned lines:

| | `hot` (old) | `removal` (new) |
|---|---|---|
| clean readings that alarm | 0.007 | 0.011 |
| noise 0.25× | 0.015 | **0.941** |
| noise 0.50× | 0.386 | **1.000** |
| feature lie 5% | 0.000 | **0.751** |
| whole-sensor lie, 1 sensor | 0.000 | **0.471** |
| whole-sensor lie, 2 sensors | 0.000 | **0.668** |
| same-gas lie, 1 sensor | 0.000 | **0.370** |

With `chi2` lines the 1-sensor lie goes from 0.036 to 0.854. When a 1-sensor lie is caught, the count is almost always exactly 1 and the first suspect is the right sensor (figure: `figures/hot_sensors_random_empirical_removal.png`).

**Limits.**
- With learned lines about half of 1-sensor lies never raise an alarm in Step 1. Clean distances have a fat tail: the 1% line is around 1,900 while a typical clean reading scores 56.
- **Sensor drift breaks it.** With `--split temporal` (learn on batches 1–7, test on 8–10) both methods catch 0% of lies with learned lines. The sensors aged so much that a clean reading already looks like a lie on one or two sensors, and the 1% line climbs above 30,000. `chi2` appears to work there, but only because it alarms on 58% of clean readings. This needs recalibration over time, not a better Step 2.

### 5.2 Industrial oven (`run_roast_spikes.py`)

**What it is.** 15 thermometers in 5 chambers, one reading a minute, about four years (`Dataset3/data_X.csv`).

**What is special.**
- The thermometers **do not move together**. Minute-to-minute moves have a correlation of about 0.03 even within the same chamber, and 0.10 over an hour. There are no witnesses to compare.
- Readings are **whole degrees**, and about 82% of minutes show no change at all.
- Movements **persist**: a rising thermometer keeps rising (autocorrelation 0.5–0.8), so `trend` removes about 40% of the usual movement.

**Results** (one-minute corruptions, learned lines):

| | `jump` + `hot` (old) | `trend` + `hot` | `trend` + `removal` |
|---|---|---|---|
| clean minutes that alarm | 0.008 | 0.009 | 0.010 |
| noise 1× usual move | 0.000 | 0.001 | **0.634** |
| noise 2× | 0.000 | 0.301 | **0.984** |
| noise 4× | 0.000 | 0.971 | **1.000** |
| obvious lie, 1 thermometer | 0.874 | **0.964** | 0.951 |
| obvious lie, 6 thermometers | **1.000** | 0.000 | 0.252 |
| plausible lie, 1 | 0.002 | 0.058 | **0.063** |
| plausible lie, 6 | 0.004 | 0.133 | **0.180** |
| stuck, 1 | 0.000 | 0.041 | **0.044** |

**The trade-off.** Realistic noise, rounded to whole degrees, changes only a few thermometers by about one degree. To catch it, the automatic cutoff for "how many is many" drops (to 3 of 15 for `trend` + `hot`), and then an obvious lie on 6 thermometers gets called noise. The old `jump` + `hot` keeps those lies at 100% but catches no noise at all. Over 10-minute windows, `trend` + `removal` catches noise at 1× in 99.6% of cases.

**Limits.** Plausible lies and stuck thermometers stay mostly invisible: under 20% at one minute and under 3% over ten minutes. A replayed stretch has its own natural trend, and with no neighbour that moves along, nobody disagrees.

### 5.3 Elder's home (`run_home_monitor.py`)

**What it is.** One person's home, 9 air sensors (temperature, humidity, two CO2, four MOX gas sensors, CO), one reading every 20 seconds from November 2019 to February 2020. A second file records which room the person enters ([UCI 799](https://archive.ics.uci.edu/dataset/799)).

**What is special.** Unlike the oven, the gas sensors **agree**: MOX2–4 move together (0.6–0.7 per step, 0.95 over ten minutes). Readings are whole numbers, and 50–88% of steps show no change.

**Results** (`jump`, learned lines; `trend` gives almost the same):

| | 1 step, `hot` | 1 step, `removal` | 30 steps, `hot` | 30 steps, `removal` |
|---|---|---|---|---|
| clean readings that alarm | 0.011 | 0.011 | 0.009 | 0.009 |
| noise 2× | 0.000 | **0.575** | 0.772 | **1.000** |
| noise 4× | 0.047 | **0.955** | 1.000 | 1.000 |
| obvious lie, 1 sensor | **0.942** | 0.929 | 0.694 | 0.672 |
| obvious lie, 4 sensors | **1.000** | 0.377 | 0.334 | **0.956** |
| plausible lie, 1 | 0.019 | 0.016 | 0.006 | 0.006 |
| stuck, 1 | 0.000 | 0.001 | 0.000 | 0.001 |

**Why a stuck sensor cannot be caught.** Most of the time the air in a home simply does not change, so a sensor frozen for ten minutes shows almost exactly what it would have shown anyway. A separate check, not part of the script, split stuck sensors by how much the sensor *really* moved during those ten minutes:

| Real movement (in standard deviations) | Share of time | Caught | Right sensor blamed, when caught |
|---|---|---|---|
| under 1 | 95% | 0.00 | – |
| 1–2 | 4% | 0.02 | 0.67 |
| 2–4 | 1% | 0.29 | 0.97 |
| over 4 | 0.2% | 0.44 | 0.94 |

A lie only shows when something is happening, and then `removal` blames the right sensor almost every time.

**Are "false" alarms really false?** Clean readings, split by whether the person entered a room in the previous 10 minutes:

| | all | nobody moved | living room | bedroom | bathroom | kitchen | hallway |
|---|---|---|---|---|---|---|---|
| `jump` + `hot` | 0.010 | 0.004 | 0.035 | 0.020 | 0.025 | **0.049** | 0.028 |
| `jump` + `removal` | 0.011 | 0.005 | 0.040 | 0.021 | 0.030 | **0.058** | 0.034 |

With nobody moving, about 1 reading in 200 alarms. Right after someone enters the kitchen, about 1 in 20. Roughly two thirds of the "false" alarms come within 10 minutes of someone moving: they are probably cooking or showering, not sensor errors.

### 5.4 Room with a head count (`run_room_monitor.py`)

**What it is.** An office room with four sensor nodes measuring temperature, light and sound, plus CO2 and two motion sensors. One reading about every 30 seconds over seven days ([UCI 864](https://archive.ics.uci.edu/dataset/864)). **Every reading says how many people (0–3) were in the room**, so this is the one dataset where we know for certain whether an alarm is a real event.

We use 13 channels. `S5_CO2_Slope` is left out because it is calculated from CO2, not measured. The two motion sensors are left out because they are on/off switches, and jitter on a switch is not physical.

**What is special.** The four nodes' lights and sounds move together (0.2–0.6), because they react to the same people. Temperature and sound flicker up and down (step autocorrelation about −0.45), which `trend` can use.

**Results** (one step, learned lines):

| | `jump` + `hot` | `jump` + `removal` | `trend` + `hot` | `trend` + `removal` |
|---|---|---|---|---|
| clean readings that alarm | 0.008 | 0.008 | 0.006 | 0.009 |
| noise 1× | 0.007 | 0.046 | 0.016 | **0.158** |
| noise 2× | 0.266 | 0.831 | 0.263 | **0.927** |
| noise 4× | 0.923 | 0.999 | 0.891 | **1.000** |
| obvious lie, 1 sensor | **0.506** | 0.468 | 0.495 | 0.492 |
| obvious lie, 4 sensors | 0.322 | 0.656 | 0.341 | **0.663** |
| plausible lie, 1 | 0.009 | 0.009 | **0.031** | 0.014 |
| stuck, 1 | 0.000 | 0.000 | 0.000 | 0.000 |

Obvious lies are only caught about half the time: the room is empty and dark most of the time, so a reading copied from another moment often looks like the real one.

**Alarms versus the true head count** (clean test readings never used for lines or the cutoff):

| | empty (1,309 readings) | occupied (124) | someone just came or went (14) |
|---|---|---|---|
| `jump` + `hot` | 0.001 | 0.056 | 0.071 |
| `jump` + `removal` | 0.000 | 0.040 | 0.071 |
| `trend` + `hot` | 0.000 | 0.056 | 0.071 |
| `trend` + `removal` | 0.000 | 0.097 | 0.071 |

In an empty room almost nothing alarms. Nearly every "false" alarm happens with people in the room, which confirms the finding from the home. The occupied samples are small (124 and 14 readings), so treat those percentages as rough.

### 5.5 Walking with and without a brace (`run_gait_monitor.py`)

**What it is.** Ten people each walked ten times in three conditions: no brace, a knee brace, and an ankle brace (both braces on the right leg). Each walk is one gait cycle of 101 points with six joint angles: ankle, knee and hip of each leg ([UCI 760](https://archive.ics.uci.edu/dataset/760)). Walking without a brace is the clean data (walks 1–7 train, 8–10 test). The braced walks are never seen until the very end.

**What is special.**
- The two legs are **near twins**: the same joint left and right moves almost in step (0.91–0.98).
- The curves are **very smooth**: `trend` predicts the next point so well that the surprise is only 1% of a normal move. The data looks smoothed and resampled to 101 points, so it is smoother than raw sensor data would be.
- A brace is **a real change that looks exactly like a lie on one joint**: the knee brace cuts the right knee's range from about 66° to 22°, and the ankle brace cuts the right ankle's from about 26° to 6°.

**Results on our corruptions** (learned lines):

| | 1 point, `jump` + `removal` | 1 point, `trend` + `removal` | 10 points, `trend` + `hot` |
|---|---|---|---|
| clean that alarm | 0.010 | 0.011 | 0.013 |
| noise 0.25× | 0.024 | **0.816** | **1.000** |
| noise 1× | 0.624 | **0.955** | **1.000** |
| obvious lie, 1 joint | 0.898 | **0.990** | **1.000** |
| plausible lie, 1 joint | 0.433 | **0.948** | **1.000** |
| stuck, 1 joint | 0.331 | **0.953** | **0.999** |

For the first time, plausible lies and stuck sensors are caught: on a smooth curve every unnatural kink stands out. Expect less on raw data.

**The braces** (detectors tuned on unbraced walking, learned lines). The best is `jump` + `removal` over 10 points:

| | alarms | called a lie | first joint set aside is the braced joint (either leg) | ...on the braced leg |
|---|---|---|---|---|
| no brace (test) | 0.012 | | | |
| knee brace | **0.506** | 0.969 | 0.780 | 0.59 (right knee) |
| ankle brace | 0.174 | 0.988 | 0.703 | |

- The monitor sees that one joint disagrees, calls it a lie, and finds **which joint** 70–78% of the time.
- It often **blames the healthy leg**. With only two twin witnesses it can see that the knees disagree, but there is no third witness to say which one is right.
- `trend`, which catches almost every artificial corruption, **barely sees the braces** (2–10% alarms). A braced knee moves less, but smoothly, so it agrees with its own recent past. You need the other leg to notice.

---

## 6. What we learned

1. **Comparing sensors with each other beats judging each one alone.** It fixed whole-sensor lies on the gas array (0% → 47–85%) and noise everywhere. But it only works when sensors normally move together. In the oven they do not, and nothing can replace a missing witness.
2. **If nothing is happening, a lie cannot be caught.** A frozen sensor in a quiet home, or a replayed quiet stretch, shows almost the true value. Lies become visible only during activity, and then the right sensor is usually named.
3. **Many "false alarms" are real life.** Cooking in the home, people in the room, a brace on a knee. A deployed system should know the context (who is where, doing what) before it raises the alarm.
4. **Two witnesses can tell you they disagree, not who is right.** With left/right twins the monitor finds the right joint but often the wrong leg. You need three or more independent views.
5. **Watching each sensor's own past and comparing sensors catch different things.** `trend` catches sudden artificial faults on smooth signals, while `removal` across sensors catches slow, consistent real changes such as a brace. A good system uses both.
6. **Test the test.** Two of our own bugs made results look better than they were: noise that produced readings no real sensor could output, and alarm lines sitting exactly on tied values. Fair scoring, resolution rounding and lines placed between values fixed them.
7. **Sensors age.** On the gas array, a monitor learned on early batches fails on later ones. Long-running monitors need recalibration.

---

## 7. Which setup should I use?

| If you care most about... | Use |
|---|---|
| A realistic false-alarm rate | `--threshold empirical` (always) |
| Catching noise | `--stage2 removal` |
| Catching lies on sensors that normally agree | `--stage2 removal` |
| Naming the guilty sensor | `--stage2 removal`. It lists the sensors it set aside |
| Catching obvious lies on *many* sensors at once | `--stage2 hot`. `removal` tends to call those noise |
| Smooth signals or movements that persist (oven, walking) | `--detector trend` |
| Slow, consistent real changes (a brace) | `--detector jump` with `--stage2 removal` |

---

## 8. Getting the data

The gas data is in the repository (`Dataset/`). The other datasets live compressed on their own branches, so the code branch stays small. Unpack the ones you need from the repository root:

| Dataset | Branch | Commands |
|---|---|---|
| Oven | `oven-data` | `mkdir -p Dataset3` then `git show origin/oven-data:Dataset3/data_X.csv.gz \| gunzip > Dataset3/data_X.csv` |
| Home | `home-data` | `mkdir -p DatasetHome` then `git show origin/home-data:DatasetHome/database_gas.csv.gz \| gunzip > DatasetHome/database_gas.csv` and `git show origin/home-data:DatasetHome/database_pos.csv.gz \| gunzip > DatasetHome/database_pos.csv` |
| Room | `room-data` | `mkdir -p DatasetRoom` then `git show origin/room-data:DatasetRoom/Occupancy_Estimation.csv.gz \| gunzip > DatasetRoom/Occupancy_Estimation.csv` |
| Walking | `gait-data` | `mkdir -p DatasetGait` then `git show origin/gait-data:DatasetGait/gait.csv.gz \| gunzip > DatasetGait/gait.csv` |

Run `git fetch origin oven-data home-data room-data gait-data` first if the branches are not in your clone yet. The unpacked folders are gitignored.

---

## 9. Running it

```
python -m pip install -r requirements.txt
```

| Command | What it runs | Time |
|---|---|---|
| `python run_workshop_repeats.py` | Gas: 2 splits × 2 line types × `hot`/`removal`, five seeds, figures | ~4 min |
| `python run_neighbor_check.py` | Gas: one quick pass, `hot` and `removal` side by side | under a minute |
| `python run_random_catch.py` | Gas: raw catch rates for each corruption | seconds |
| `python run_full_vs_blocks.py` | Gas: 16 per-sensor tests vs one test on all 128 numbers, with timings | seconds |
| `python run_roast_spikes.py` | Oven: 1 and 10 minutes × `jump`/`trend` × line types × `hot`/`removal` | ~5 min |
| `python run_home_monitor.py` | Home: same grid, then alarms by room | ~9 min |
| `python run_room_monitor.py` | Room: same grid, then alarms by head count | ~1 min |
| `python run_gait_monitor.py` | Walking: same grid, then the braced walks | ~30 s |

Every time-series script accepts the same options to run just part of the grid, for example:

```
python run_roast_spikes.py --minutes 1 --detector trend --threshold empirical --stage2 removal
python run_home_monitor.py --steps 1 --detector jump --threshold empirical
python run_workshop_repeats.py --split random --threshold empirical --stage2 removal
```

`--target` changes the clean false-alarm rate for learned lines (default 0.01).

**Reading the output.** Each configuration prints the cutoff chosen for each seed, then a table with the min, median and max over the five seeds. A side-by-side table of medians follows. For the time-series scripts, the share of corruptions that actually changed the reading comes first: anything below 1.0 means some corruptions were invisible by construction and were left out.

**Tests.**

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

42 tests. Most use small synthetic data and run in a few seconds. The data-loader tests skip themselves when `Dataset/` is missing.

---

## 10. What is in the repository

| File | What it does |
|---|---|
| `monitor.py` | `BlockMahalanobisMonitor` (one test per sensor block) and `FullMahalanobis` (one test on all numbers, with standardized features) |
| `consistency.py` | `SensorRemovalCheck`: the "set sensors aside until the rest agree" count, with its own chi-square or learned lines |
| `thresholds.py` | Learned alarm lines: one line, or jointly tuned per-sensor lines, placed between clean values |
| `corruptions.py` | Gas corruptions: noise, feature lies, whole-sensor lies (optionally same-gas) |
| `timeseries.py` | Shared machinery for sensors read over time: windows, `jump`/`trend` surprises, corruptions, resolution rounding, fair scoring, the five-seed experiment |
| `data_loader.py` | Loads the gas batches; random and time-ordered splits |
| `run_*.py` | One script per experiment (see [section 9](#9-running-it)) |
| `tests/` | Unit tests |
| `figures/` | Histograms of how many sensors are counted for clean, noisy and lying readings (gas) |

---

## 11. History of changes

1. **Original.** Gas and oven, chi-square lines, `hot` count. The original reported cutoffs (10 of 16 on gas, 5 of 15 on the oven) and catch rates come from this version and are superseded.
2. **Fairer tests.** The cutoff is tuned with the exact scoring rule. Oven noise is sized against the usual one-minute move. Time-ordered gas split, learned lines, whole-sensor and same-gas lies added.
3. **The witness check.** `removal` added after finding that whole-sensor lies were invisible to `hot`. Features standardized (covariance condition number about 1e14 → 1e8), distances vectorized (gas run 70 s → 28 s), shared code moved into modules, tests added.
4. **Oven rework.** Fair scoring (unchanged readings are left out), the `trend` detector, 10-minute replayed and stuck lies, and the gas monitors reused on surprise windows.
5. **Home.** Shared time-series code moved into `timeseries.py`; home monitor and alarms-by-room analysis.
6. **Room and two test fixes.** Room monitor with ground-truth head count. Noise is rounded to each sensor's resolution, and learned lines are placed between clean values rather than on a tied value. All tables were rerun after these fixes.
7. **Walking.** Gait monitor and the brace analysis.
