# PDCRL paper code and synthetic data

This is the public code for *PDCRL: Process-Driven Curriculum Reinforcement Learning for Large-Scale Hot-Rolling Batch Scheduling*. It contains the executable method implementations, frozen paper configuration, synthetic training/development/test batches, and the checkpoints used for the reported evaluation.

## Data

The release implements the paper's four-objective HPCVRP formulation:

- process-transition cost from directional width, gauge, and hardness penalties;
- the prize penalty for slabs postponed from the current batch;
- adjacent specific-power change;
- `1000 × number of rolling units` roll-change cost.

The included data split is exact and disjoint at each of p151, p271, p331, p600, p1000, and p2000. Each role has a separate public location:

- `data/training/instances`: 1,131 training CSVs
- `data/validation/instances`: 168 deterministic nested subsets derived from three development parents per scale for checkpoint selection;
- `data/test/instances`: 60 independently generated test batches, 10 per scale.

`data/training/manifest.json`, `data/validation/manifest.json`, and `data/test/manifest.json` record every generator seed, the nested-subset seed namespace, the process-profile hash, and the expected CSV hashes for the three disjoint splits.

### Regenerate every dataset from code

Choose a path that does not yet exist:

```bash
python run.py generate-data --output regenerated_data
```

This regenerates all 1,131 training, 168 validation, and 60 test CSVs from the public seeds and namespace. It verifies every SHA-256 before making `regenerated_data` visible. The command never writes to the bundled `data/` directory and refuses to run when the requested output path already exists.

## Installation and integrity check

Python 3.10 and PyTorch 2.5 were used for the reported runs. A CUDA GPU is strongly recommended for learned methods.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python run.py verify
```

`verify` checks all three manifests, every CSV and checkpoint hash, split disjointness, expected
record counts, and the frozen process profile.

## One-script usage

Evaluate the raw frozen PDCRL checkpoint on the first p151 test batch:

```bash
python run.py evaluate --method pdcrl --scale 151 --replicate 0 --raw
```

Evaluate complete PDCRL with the paper's bounded local search:

```bash
python run.py evaluate --method pdcrl --scale 151 --replicate 0
```

Evaluate another learned method. `direct` is reported only at p1000/p2000;

```bash
python run.py evaluate --method pomo --scale 151 --replicate 0
python run.py evaluate --method direct --scale 1000 --replicate 0
python run.py evaluate --method scale_only --scale 1000 --replicate 0
```

Run a native baseline. The algorithm seed is fixed to the test-batch replicate, as in the paper.

```bash
python run.py baseline --method greedy --scale 151 --replicate 0
python run.py baseline --method grasp --scale 151 --replicate 0
python run.py baseline --method nsga2 --scale 151 --replicate 0
python run.py baseline --method alns --scale 151 --replicate 0
```

GRASP uses 30 restarts and up to 60 s of deterministic local search per restart. NSGA-II uses a population of 200 for 2,000 generations. ALNS checks every 1,000 iterations and uses the actual paper-result stopping rule: five consecutive segments below 0.1% scalar improvement. These methods can take minutes to tens of hours at the largest scale. ALNS writes a resumable state.

Retrain one checkpoint with the frozen wall-clock protocol, or first run a short mechanics smoke
test:

```bash
python run.py train --method pdcrl --scale 1000 --device cuda
python run.py train --method pdcrl --scale 151 --device cpu --smoke
```

Outputs are written under `artifacts/`, which is ignored by Git. Full wall-clock retraining is hardware-sensitive; exact numerical verification should start from the supplied immutable checkpoint, while retraining verifies the released optimization protocol.

## Paper-value cross-check

The following means over the same 10 test batches are the frozen paper reference values (lower is better). The supplied checkpoints reproduce the learned constructors; wall-clock-bounded local search and native solvers can reach slightly different endpoints on different hardware:

| Method | p151 | p271 | p331 | p600 | p1000 | p2000 |
|---|---:|---:|---:|---:|---:|---:|
| Greedy | 52,261.1 | 75,662.9 | 87,777.3 | 122,699.0 | 162,715.7 | 238,443.1 |
| GRASP | 44,126.2 | 64,608.7 | 75,558.2 | 119,357.3 | 189,269.4 | 317,323.7 |
| NSGA-II | 48,392.2 | 73,986.8 | 86,632.3 | 121,257.5 | 161,365.7 | 236,199.8 |
| ALNS | 39,674.0 | 60,123.2 | 68,887.4 | 100,008.9 | 132,835.2 | 197,372.9 |
| POMO | 48,255.5 | 63,452.9 | 112,173.6 | 91,298.7 | 167,015.1 | 299,987.1 |
| PDCRL without LS | 41,384.6 | 58,397.5 | 65,382.9 | 88,904.7 | 113,960.1 | 178,679.8 |
| PDCRL | 38,546.4 | 55,274.7 | 61,707.2 | 81,566.3 | 104,247.8 | 171,487.6 |

The ablation table uses Direct RL `128,300.3 / 200,214.4`, scale-only
`137,917.6 / 252,321.1`, and PDCRL without LS `113,960.1 / 178,679.8` at
p1000/p2000.

## Files retained by dependency, but not directly used by paper
Note that some files or code segments have been retained due to dependencies, even though they are not directly invoked in the method described in the paper.


## Licensing

This release is MIT licensed. The POMO baseline is a problem-adapted independent implementation; no source code from the upstream POMO repository is redistributed.
