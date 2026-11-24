# Sensors: noise or a lie?

When a group of sensors starts reporting strange values, there are two explanations.

- **Noise.** Every sensor gets a little shaky at once, like static on a radio.
- **A lie.** One or two sensors report wrong values while the rest are fine. The sensor might be broken, stuck, or tampered with.

This repository builds a monitor that first notices *that* something is wrong, then decides *which* of the two it is. It is tested on five real datasets: a gas-sensor array, an industrial oven, an elderly person's home, an office room with a head count, and people walking with and without a leg brace.

Every monitor learns only from clean data. Nothing is tuned by hand, and every reported number comes from data the monitor never saw while it was being set up.

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

The monitor asks one question: **how many witnesses do I have to ignore before everyone else agrees?**

| Witnesses ignored | Verdict |
|---|---|
| 0 | Nothing is wrong (no alarm) |
| 1 or 2 | **A lie**, and the ignored ones are the suspects |
| Most of them | **Noise** |

This only works when the sensors normally agree with each other. That is the single most important fact in this project, and it explains most of the results below.

---

## 2. How the monitor works

### Step 1: is anything wrong?

The monitor learns what normal looks like from clean data: the average reading and how readings normally move together. For a new reading it computes one number, the **Mahalanobis distance**. Think of it as "how surprised should I be?". Breaking the usual relationships between sensors counts as much as an odd value on its own. If the distance is above a line, the alarm goes off.

### Step 2: noise or lie?

Two ways to count how many sensors are involved:

| Name | What it counts | Strength | Weakness |
|---|---|---|---|
| `hot` | Sensors whose own readings look odd on their own | Simple | Blind to a sensor that shows a real-looking value from another moment, because on its own it looks fine |
| `removal` | Sensors that must be **set aside** before the rest agree | Sees sensors that disagree with each other, and **names the suspects** | Needs sensors that normally move together |

A **cutoff** then decides: a count below the cutoff is a lie, a count at or above it is noise. The cutoff is chosen automatically (see [section 3](#3-how-we-test-it)).

`removal` does not refit anything. One identity splits the distance into "what the other sensors explain" plus "what sensor *k* adds on its own" (`d = d_without_k + w_kᵀ (P_kk)⁻¹ w_k`, with `P` the inverse covariance and `w = P (x − μ)`). Ranking the sensors is one matrix–vector product plus one small quadratic form per sensor.

### For sensors read over time: watch the surprise, not the value

In four of the five datasets each sensor is read repeatedly. A value on its own says little: the oven is hot during the day and cool at night. What matters is **how the reading moved since last time**, compared with how it usually moves.

| Name | Expected move | Good when |
|---|---|---|
| `jump` | The average move | Readings move randomly |
| `trend` | A prediction from the sensor's own last 3 moves | A rising temperature keeps rising |

The **surprise** is the actual move minus the expected move. The monitor watches the last *W* surprises of each sensor (*W* = 1, or a longer window such as 10 minutes) and applies the same two steps.

### Where the alarm lines come from

| Name | How the line is set | Note |
|---|---|---|
| `chi2` | A textbook formula that assumes bell-curve data | Real data has fatter tails, so it usually rings far too often: about 8–11% of clean readings on gas and oven, and up to 58% on drifted gas data, instead of the intended 0.1% |
| `empirical` | Learned from clean data so that about **1%** of clean readings cross it | Honest false-alarm rate. **Use this one.** |

---

## 3. How we test it

Clean, real data is broken on purpose in known ways. The monitor has to notice, and to name the right kind of problem.

The four time-series datasets (oven, home, room, gait) run the same four attacks. The gas array is a stack of snapshots, so two of those attacks have a different shape there, and one is not run.

| Attack | Oven, home, room, gait | Gas array |
|---|---|---|
| **noise** | Jitter on every channel, scaled to that channel's usual step | Jitter on every feature, scaled to that feature's standard deviation. Same scales, 0.25× through 4× |
| **obvious lie** | 1 or more channels overwritten with a reading from another moment | All 8 numbers of 1 or more sensors copied from another clean row |
| **plausible lie** | 1 or more channels replay a normal-looking stretch of steps | The same copy, but the donor row is the same gas |
| **stuck** | 1 or more channels freeze at their last value | The sensor is frozen at the previous snapshot |
| **feature lie** | Not run | 5%, 10% or 25% of single numbers, scattered across sensors |

**What "correct" means.** A noisy reading is correct if it is called noise. A lie is correct if it raises an alarm *and* is called a lie. Every number in the tables is the share of corrupted readings that were called the right kind.

**Fair scoring.** Readings that already alarmed before the corruption are left out. So are corruptions that did not change the stored value. In the oven, readings are whole degrees and the temperature usually does not move from one minute to the next, so 61% of one-thermometer replays left the value exactly as it was. Noise is rounded to each sensor's own resolution.

**No peeking.** Each corpus is split in time.

1. **Training** (usually the first 70%): learns what normal looks like.
2. **Calibration** (half of the rest): sets the learned lines and chooses the noise-versus-lie cutoff.
3. **Evaluation** (the other half): produces every number below.

Five shuffles (seeds 42–46) repeat the calibration/evaluation cut. Tables show the median.

Every corpus was scored on the same grid: noise at 0.25×, 0.50×, 1×, 2× and 4×, and lies on 1, 2 and 4 channels. The tables below keep the rows used to compare the datasets.

---

## 4. Results at a glance

Learned (`empirical`) lines, about 1% false alarms, median of seeds 42–46. One row per dataset, for the setup that does best on that dataset. The noise column is the rate at the scale in the previous column: those scales are not the same, so the five noise rates are not one comparison. The three lie columns are the one-sensor case.

| Dataset | Sensors agree? | Best setup | Noise scale | Noise | Obvious lie, 1 | Plausible lie, 1 | Stuck, 1 |
|---|---|---|---|---|---|---|---|
| Gas array | Yes, but the hold-out has drifted | `removal` | 0.25× | 0.50 | 0.09 | 0.07 | 0.09 |
| Oven | No | `trend` + `removal`, 1 minute | 2× | 0.99 | 0.95 | 0.06 | 0.04 |
| Home | Partly (the gas sensors) | `jump` + `removal`, 1 step | 4× | 0.96 | 0.93 | 0.02 | 0.00 |
| Room | Partly (light and sound) | `trend` + `removal`, 1 step | 2× | 0.93 | 0.49 | 0.01 | 0.00 |
| Walking | Yes (left and right) | `trend` + `removal`, 1 point | 1× | 0.96 | 0.99 | 0.95 | 0.95 |

How to read it:

- **Noise is caught once it is large enough for that corpus.** With `removal`, that point is 0.50× on the drifted gas array (0.25× is only 0.50), 1× while walking, 2× on the oven and in the room, and 4× on a single home step.
- **An obvious one-sensor lie is caught where a neighbour can object:** gait 0.99, oven 0.95, home 0.93, room 0.49. On the drifted gas hold-out it is 0.09.
- **A replay and a stuck sensor are caught while walking (0.95) and almost nowhere else.** The oven sees them at 0.06 and 0.04. Home, room and the gas array stay at or under 0.09.

---

## 5. The five datasets

### 5.1 Gas sensor array (`run_workshop_repeats.py`)

**What it is.** 16 chemical sensors, 8 numbers each (128 per reading), 13,910 readings in 10 batches over about three years (UCI Gas Sensor Array Drift). Each reading is a snapshot, so there is no `jump` or `trend`. The monitor scores the values themselves.

**What is different.** A lying sensor copies a real reading, so its 8 numbers look ordinary on their own. `hot` finds no odd sensor and the alarm is thrown away.

**The result.** Earliest 70% of readings for training, last 30% held out, in recording order (`--split ordered`). The hold-out is the end of batch 8 and batches 9–10, after the sensors have drifted. Learned lines, median of seeds 42–46.

| | `hot` | `removal` |
|---|---|---|
| clean readings that alarm | 0.006 | 0.011 |
| noise 0.25× | 0.006 | **0.501** |
| noise 0.50× | 0.021 | **1.000** |
| obvious lie, 1 sensor | 0.000 | 0.086 |
| obvious lie, 2 sensors | 0.000 | 0.125 |
| plausible lie, 1 sensor (same gas) | 0.000 | 0.065 |
| stuck, 1 sensor | 0.000 | 0.088 |
| feature lie, 5% | 0.000 | 0.160 |

About 0.095 of the one-sensor obvious lies cross the distance threshold. On `removal`, all five seeds set the cutoff to 2, so one sensor set aside is called a lie and that rate is 0.086. When the lie is caught, the count is exactly 1 and the first sensor named is the lying one, both at 1.000. The 1% line, set on the calibration half of these late rows, sits near 31,000. A typical training score is about 58, and a typical held-out score is about 247. `removal` detects the injected noise at 0.501 for 0.25× and at 1.000 from 0.50× up. `hot` still scores every lie at 0.000. Figure: `figures/hot_sensors_ordered_empirical_removal.png`.

Stopping at the batch boundary (`--split temporal`: train on batches 1–7, test on 8–10) labels 0.077 of the one-sensor obvious lies, at the median, with the same cutoff of 2. `chi2` appears to detect them only because it alarms on 58% of clean late readings. The array needs a fresh calibration, which is an offline refit, not a larger online kernel.

### 5.2 Industrial oven (`run_roast_spikes.py`)

**What it is.** 15 thermometers in 5 chambers, one reading a minute, about four years.

**What is different.** The thermometers do not move together. Minute-to-minute correlation is about 0.03 even inside one chamber. Readings are whole degrees, and about 82% of minutes do not change. Movements persist (autocorrelation 0.5–0.8), so `trend` removes about 40% of the ordinary step.

**The result.** One minute, learned lines. Columns are the kernels that this comparison uses.

| | `jump` + `hot` | `trend` + `hot` | `trend` + `removal` |
|---|---|---|---|
| clean minutes that alarm | 0.008 | 0.009 | 0.010 |
| noise 1× | 0.000 | 0.000 | **0.699** |
| noise 2× | 0.000 | 0.034 | **0.992** |
| noise 4× | 0.000 | 0.792 | **1.000** |
| obvious lie, 1 | 0.877 | **0.966** | 0.947 |
| obvious lie, 4 | **0.999** | **1.000** | 0.572 |
| plausible lie, 1 | 0.002 | 0.060 | **0.064** |
| plausible lie, 4 | 0.003 | 0.103 | **0.123** |
| stuck, 1 | 0.000 | 0.032 | **0.042** |

`trend` + `removal` is the first kernel that sees realistic noise: 0.699 at one times the usual step and 0.992 at two times. `trend` + `hot` sets the cutoff at 5 of 15, so four thermometers stay a lie (1.000). The same four under `trend` + `removal` fall to 0.572, because that many are called noise. `jump` + `hot` keeps the four-thermometer lie at 0.999 and detects no noise at these scales. Over ten minutes, `trend` + `removal` detects noise at 1× in 1.000 of cases.

A one-thermometer replay stays near 0.06 and a stuck thermometer near 0.04 at one minute. At ten minutes both are under 0.01. A replayed stretch carries its own trend, and no neighbouring thermometer moves with it.

### 5.3 Elder's home (`run_home_monitor.py`)

**What it is.** One apartment, 9 air channels (temperature, humidity, two CO₂ sensors, four metal-oxide sensors, CO), one reading every 20 seconds from November 2019 to February 2020 ([UCI 799](https://archive.ics.uci.edu/dataset/799)). A second file records which room the resident enters. That file is never given to the detector.

**What is different.** The metal-oxide channels agree (0.6–0.7 per step, about 0.95 over ten minutes). Readings are whole numbers, and 50–88% of steps do not change. `trend` gives almost the same rates as `jump`, so the table is `jump` only, at the same two windows the other time-series sections use: one step, and a long window.

| | 1 step, `hot` | 1 step, `removal` | 30 steps, `hot` | 30 steps, `removal` |
|---|---|---|---|---|
| clean readings that alarm | 0.011 | 0.011 | 0.009 | 0.009 |
| noise 2× | 0.000 | **0.575** | 0.772 | **1.000** |
| noise 4× | 0.047 | **0.955** | **1.000** | **1.000** |
| obvious lie, 1 | **0.942** | 0.929 | **0.694** | 0.672 |
| obvious lie, 4 | **1.000** | 0.377 | 0.334 | **0.956** |
| plausible lie, 1 | 0.019 | 0.016 | 0.006 | 0.006 |
| stuck, 1 | 0.000 | 0.001 | 0.000 | 0.001 |

At one step, `removal` is what catches noise, and only at 4× (0.96). An obvious lie on one sensor is 0.93 either way. On four sensors the counts trade places: `hot` stays at 1.00 and `removal` falls to 0.38, because that many disagreeing sensors are called noise. Over 30 steps (10 minutes) the same noise at 2× is 0.77 on `hot` and 1.00 on `removal`, and the four-sensor lie flips the other way (`hot` 0.33, `removal` 0.96).

A stuck channel over 30 steps is almost never visible, because the air is constant on most steps and the frozen value matches the true one.

**Are the clean alarms false?** This check is one even/odd split of the hold-out, not the five-seed median above. It asks whether an alarm on unmodified data falls within 10 minutes of the resident entering a room.

| | all | nobody moved | living room | bedroom | bathroom | kitchen | hallway |
|---|---|---|---|---|---|---|---|
| `jump` + `hot` | 0.010 | 0.004 | 0.035 | 0.020 | 0.025 | **0.049** | 0.028 |
| `jump` + `removal` | 0.011 | 0.005 | 0.040 | 0.021 | 0.030 | **0.058** | 0.034 |

With nobody moving, about one clean sample in 200 alarms. After an entry to the kitchen, about one in 20. Roughly two thirds of the clean alarms fall within ten minutes of the resident moving.

### 5.4 Office room (`run_room_monitor.py`)

**What it is.** One office on seven dates, 10,129 readings, about one every 31 seconds ([UCI 864](https://archive.ics.uci.edu/dataset/864)). Four nodes measure temperature, light and sound. CO₂ is kept. The CO₂ slope is left out because it is derived, and the two motion channels are left out because they are binary switches. 13 channels remain. Every row carries the number of people in the room, from 0 to 3. That label is never used for fitting.

**What is different.** Light and sound at the four nodes respond to the same occupants, so those channels normally agree. The room is empty and dark on most readings.

**The result.** One step, learned lines. This is the one time-series table that shows both residuals and both counts.

| | `jump` + `hot` | `jump` + `removal` | `trend` + `hot` | `trend` + `removal` |
|---|---|---|---|---|
| clean readings that alarm | 0.008 | 0.008 | 0.006 | 0.009 |
| noise 1× | 0.007 | 0.046 | 0.016 | **0.158** |
| noise 2× | 0.266 | 0.831 | 0.263 | **0.927** |
| noise 4× | 0.923 | 0.999 | 0.891 | **1.000** |
| obvious lie, 1 | **0.506** | 0.468 | 0.495 | 0.492 |
| obvious lie, 4 | 0.322 | 0.656 | 0.341 | **0.663** |
| plausible lie, 1 | 0.009 | 0.009 | **0.031** | 0.014 |
| stuck, 1 | 0.000 | 0.000 | 0.000 | 0.000 |

`trend` + `removal` detects noise at 2× with rate 0.93. An obvious lie on one channel is about half, under every kernel, counting only copies that change the stored value. Many copied readings equal the live one and those are excluded. Replays stay at or below 0.031. Stuck faults stay at 0.00.

**Alarms versus the true head count.** Clean evaluation samples, never used for the lines or the cutoff. Median of seeds 42–46.

| | empty (1,304 readings) | occupied (126) | someone just came or went (20) |
|---|---|---|---|
| `jump` + `hot` | 0.001 | 0.058 | 0.071 |
| `jump` + `removal` | 0.000 | 0.075 | 0.100 |
| `trend` + `hot` | 0.000 | 0.056 | 0.050 |
| `trend` + `removal` | 0.000 | 0.095 | 0.150 |

The empty room produces essentially no alarms. Nearly every clean alarm happens with people in the room. The occupied counts are small, so those two rates are coarse. The empty-room rate is not.

### 5.5 Walking (`run_gait_monitor.py`)

**What it is.** Ten participants, ten walks each, in three conditions: no brace, a knee brace, and an ankle brace, both braces on the right leg ([UCI 760](https://archive.ics.uci.edu/dataset/760)). Each walk is one gait cycle of 101 points and six joint angles. Unbraced walks 1–7 train. Walks 8–10 are the hold-out: half sets the lines and the cutoff, half is scored. Braced walks are never used to set a threshold.

**What is different.** Left and right joints of the same type move almost in step (0.91–0.98). `trend` leaves a surprise of about 1% of a normal step. The curves look smoothed, so these rates are an upper envelope for unsmoothed motion.

| | 1 point, `jump` + `removal` | 1 point, `trend` + `removal` | 10 points, `trend` + `hot` |
|---|---|---|---|
| clean that alarm | 0.010 | 0.011 | 0.013 |
| noise 0.25× | 0.024 | **0.816** | **1.000** |
| noise 1× | 0.624 | **0.955** | **1.000** |
| obvious lie, 1 | 0.892 | **0.990** | **1.000** |
| plausible lie, 1 | 0.433 | **0.948** | **1.000** |
| stuck, 1 | 0.331 | **0.953** | **0.999** |

On one point, `trend` + `removal` detects noise at 0.25× (0.82), an obvious lie (0.99), a replay (0.95) and a stuck joint (0.95). The same faults over ten points, under `trend` + `hot`, are saturated. This is the only corpus on which a replay and a stuck channel are visible, because the two legs agree and the curves are smooth.

**The braces.** Held out from every threshold. The kernel that sees them is `jump` + `removal` over ten points. A brace shortens the range smoothly (right knee from about 66° to 22°, right ankle from about 26° to 6°), so each joint still agrees with its own recent past.

| | alarms | called a lie | first joint is the braced joint, either leg | on the braced leg |
|---|---|---|---|---|
| no brace (test) | 0.012 | — | — | — |
| knee brace | **0.506** | 0.953 | 0.780 | 0.59 (right knee) |
| ankle brace | 0.174 | 0.977 | 0.702 | — |

Of the knee-brace alarms, 0.95 are labeled a lie and 0.78 name a knee. The ankle brace is detected less often (0.17). When it is detected, it is labeled a lie in 0.98 of alarms and attributed to an ankle in 0.70. With only two twin witnesses the monitor can see that the knees disagree, and it often names the healthy leg. `trend` alarms on 0.02–0.10 of braced cycles: a braced joint moves less, but smoothly, so it agrees with its own past.

---

## 6. What we learned

1. **The same attack is a different problem on each corpus.** At a low false-alarm rate, `removal` detects injected noise on all five, at a scale from 0.50× to 4× the usual variation. An obvious one-sensor lie is caught on the oven, at home and while walking, about half the time in the office, and on 0.09 of the drifted gas hold-out.
2. **A replay or a stuck sensor needs a witness.** They are exposed where the channels are strongly correlated and the curves are smooth (gait, about 0.95). Where channels move independently, as in the oven, a consistency check has nobody to consult. In a quiet home the frozen value often equals the true one, so there is nothing to see.
3. **How many sensors deviate says where the anomaly is, not why.** One or two sensors set aside means the problem is local: a hardware fault and a deliberate substitution look the same. A count at the noise cutoff can be interference, a real event that shifts every sensor, or a coordinated change. The count should start an investigation, not close one.
4. **Many clean alarms are real life.** In the home, about two thirds of clean alarms fall within ten minutes of the resident moving. In the office, the empty room barely alarms and the occupied room accounts for the rest. A brace on a knee is a real change that the monitor calls a lie, and 0.78 of those alarms name a knee.
5. **Drift invalidates a static calibration.** On the gas array the monitor is fit on the earliest 70% of readings, and the 1% line set on the drifted hold-out sits near 31,000. That needs a periodic offline refit of the mean and the precision, not a larger online kernel.
6. **The two counts catch different widths.** `removal` catches diffuse noise and names one or two suspects. `hot` keeps an obvious lie that covers many sensors, which `removal` starts to call noise: four thermometers on the oven stay at 0.999 under `jump` + `hot` and fall to 0.572 under `trend` + `removal`.

---

## 7. Which setup should I use?

| If you care most about... | Use |
|---|---|
| A realistic false-alarm rate | `--threshold empirical` |
| Catching noise | `--stage2 removal` |
| Catching a lie on sensors that normally agree | `--stage2 removal` |
| Naming the suspect | `--stage2 removal`. It lists the sensors it set aside |
| Catching an obvious lie on many sensors at once | `--stage2 hot` |
| Smooth signals, or motion that persists (oven, walking) | `--detector trend` |
| A slow, consistent real change (a brace) | `--detector jump` with `--stage2 removal` |

---

## 8. Getting the data

The gas data is in the repository (`Dataset/`). The other datasets live compressed on their own branches. Unpack the ones you need from the repository root:

| Dataset | Branch | Commands |
|---|---|---|
| Oven | `oven-data` | `mkdir -p Dataset3` then `git show origin/oven-data:Dataset3/data_X.csv.gz \| gunzip > Dataset3/data_X.csv` |
| Home | `home-data` | `mkdir -p DatasetHome` then `git show origin/home-data:DatasetHome/database_gas.csv.gz \| gunzip > DatasetHome/database_gas.csv` and `git show origin/home-data:DatasetHome/database_pos.csv.gz \| gunzip > DatasetHome/database_pos.csv` |
| Room | `room-data` | `mkdir -p DatasetRoom` then `git show origin/room-data:DatasetRoom/Occupancy_Estimation.csv.gz \| gunzip > DatasetRoom/Occupancy_Estimation.csv` |
| Walking | `gait-data` | `mkdir -p DatasetGait` then `git show origin/gait-data:DatasetGait/gait.csv.gz \| gunzip > DatasetGait/gait.csv` |

Run `git fetch origin oven-data home-data room-data gait-data` first if those branches are not in the clone yet. The unpacked folders are gitignored.

---

## 9. Running it

```
python -m pip install -r requirements.txt
```

| Command | What it runs | Time |
|---|---|---|
| `python run_workshop_repeats.py --split ordered --threshold empirical` | Gas, earliest 70%, learned lines, `hot` and `removal` | a few minutes |
| `python run_roast_spikes.py --threshold empirical` | Oven, 1 and 10 minutes, `jump`/`trend`, `hot`/`removal` | ~5 min |
| `python run_home_monitor.py --threshold empirical` | Home, same grid, then alarms by room | ~9 min |
| `python run_room_monitor.py --threshold empirical` | Room, same grid, then alarms by head count | ~1 min |
| `python run_gait_monitor.py --threshold empirical` | Walking, same grid, then the braced walks | ~30 s |

`--target` changes the clean false-alarm rate for learned lines (default 0.01).

Each configuration prints the cutoff for every seed, then the min, median and max. A side-by-side table of medians follows. For the time-series scripts, the share of corruptions that actually changed the reading is printed first. Anything below 1.0 was invisible by construction and was left out of the rate.

**Tests.**

```
python -m pip install -r requirements-dev.txt
python -m pytest
```

---

## 10. What is in the repository

| File | What it does |
|---|---|
| `monitor.py` | `BlockMahalanobisMonitor` (one test per sensor) and `FullMahalanobis` (one test on all numbers) |
| `consistency.py` | `SensorRemovalCheck`: set sensors aside until the rest agree |
| `thresholds.py` | Learned alarm lines, placed between clean values |
| `corruptions.py` | Gas corruptions: noise, feature lies, whole-sensor lies, same-gas lies |
| `timeseries.py` | Windows, `jump`/`trend`, noise, obvious lies, replays, stuck faults, fair scoring, the five-seed experiment |
| `data_loader.py` | Gas batches, with a shuffled split, an earliest-70% split, and a batch split |
| `run_*.py` | One script per experiment |
| `tests/` | Unit tests |
| `figures/` | How many sensors are counted for clean, noisy and lying gas readings |

---

## 11. History of changes

1. **Original.** Gas and oven, chi-square lines, `hot` count.
2. **Fairer tests.** The cutoff is tuned with the scoring rule used at evaluation. Learned lines, whole-sensor lies and same-gas lies.
3. **The witness check.** `removal`, so a sensor that looks ordinary on its own can still be set aside.
4. **Oven rework.** Fair scoring, the `trend` residual, and the same monitors on surprise windows.
5. **Home, room, walking.** The shared time-series grid, alarms checked against room entry, head count, and held-out braces.
6. **This write-up.** The five result tables use the same attack names. Cells a dataset does not run are left blank, instead of a different metric being inserted in their place.
