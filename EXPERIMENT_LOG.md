# Experiment Log — Bomberman DDQN

This is the factual source for the final scientific report. Entries distinguish
measured evidence from interpretation. Unknown values are explicitly marked
**not recorded**. Preserve failed experiments.

## Current Model Architecture

- Agent: `agent_code/ddqn_agent`; CPU-only, in-process DDQN implementation.
- Action order: `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB` (six Q-values).
- Online Q-network and target Q-network: selected E35 uses a dueling MLP
  `509 → 256 → 128`, then a scalar value head and six-action advantage head;
  Q-values are `V + A - mean(A)`. The earlier standard MLP was
  `509 → 256 → 128 → 6`, with ReLU after each hidden layer.
- The online network selects the next legal action; the target network evaluates
  that action (Double DQN target).
- Checkpoints: the packaged `ddqn_model.pt` is selected E35, a 509-feature
  dueling checkpoint. `DDQN_MODEL_FILE` and `DDQN_ARCHITECTURE` permit explicit
  reproducible selection; legacy checkpoints without architecture metadata are
  interpreted as standard MLP checkpoints. E25 remains preserved as the
  conservative standard-DDQN fallback. Checkpoints store online-network
  weights, feature dimension, action ordering, and architecture; they do **not**
  store replay memory, optimizer state, target-network state, or exploration
  progress.

## State Representation

`state_to_features(game_state)` returns a float32 vector of length 509, or
`None` for terminal input.

| Component | Values | Transformation | Purpose |
|---|---:|---|---|
| Terrain | 81 | 9×9 local view centred on agent; wall/free/crate values `-1/0/1` | Nearby traversability and crates |
| Collectable coins | 81 | Binary 9×9 local map | Local reward targets |
| Opponents | 81 | Binary 9×9 local map | Opponent locations |
| Bomb timers | 81 | Bomb timer / 4 at bomb tiles | Time-to-explosion signal |
| Bomb danger | 81 | Blast-line urgency from timers, respecting stone-wall stopping | Anticipate hazardous tiles |
| Explosions | 81 | `explosion_map / 2` local view | Current immediate danger |
| Global extras | 3 | Normalized agent x/y and bomb-available flag | Absolute position and bomb capability |
| Global coin navigation | 4 | Direction x/y, shortest-path distance, and reachable-coin flag | Guide search toward coins outside 9×9 view |
| Global crate navigation | 4 | Direction x/y, shortest-path distance, and reachable-crate flag | Guide search toward free tiles adjacent to crates |
| Legality mask | 6 | Legal actions under the framework's current occupancy rules | Prevent selection of invalid actions |
| Immediate action safety | 6 | One indicator per action | Active-explosion and next-step bomb-blast awareness |

The 9×9 view is padded outside the board. Coin direction and distance are found
by BFS over free tiles. The legality mask does not encode future safety: moves
into a future blast remain available to the learned policy.

## Reward Design

Current event-based shaped reward used during training:

| Event | Reward |
|---|---:|
| `COIN_COLLECTED` | +5.00 |
| `CRATE_DESTROYED` | +1.00 |
| `KILLED_OPPONENT` | +10.00 |
| `SURVIVED_ROUND` | +2.00 |
| Immediate move reversal (E12 active) | -0.20 |
| Second and later consecutive `WAIT` actions (E28 active) | -0.20 each |
| `WAITED` | -0.02 |
| `INVALID_ACTION` | -1.00 |
| `KILLED_SELF` | -12.00 |
| `GOT_KILLED` | -12.00 |

Historical changes:

- **E01–E03:** `WAITED = -0.02`; no distance-progress reward.
- **E04:** `WAITED = -0.10`; add `+0.10` for a
  one-step reduction in shortest-path distance to the nearest reachable coin,
  and `-0.10` for a one-step increase. Distance shaping is skipped on coin
  collection so the next, farther coin cannot reduce the collection reward.

The active configuration was restored to the E01 reward after E04's matched
evaluation reduced coins per round. `USE_COIN_PROGRESS_SHAPING = False` is kept
in code so E04 can be reproduced deliberately.

## Training Method

- Replay: preallocated NumPy ring buffer, capacity 25,000 transitions.
- Batch size: 64. Learning begins after 1,000 stored transitions.
- Update schedule: one gradient update every four environment transitions.
- Discount factor: 0.99. Optimizer: Adam, learning rate `1e-4`.
- Loss: Huber (`SmoothL1Loss`). Gradient norm clipping: 1.0.
- Target-network synchronization: every 1,000 environment transitions.
- Exploration: epsilon-greedy over legal actions; linear schedule 1.00 → 0.05
  over 30,000 stored transitions.
- Terminal handling: a terminal transition is stored only when death prevents
  the framework's normal per-step callback; surviving final states are not
  stored twice.
- Checkpointing: weights and compatibility metadata are saved at every round.
- Training duration: not recorded unless an experiment entry states otherwise.

## Key Design Decisions

- Use a small MLP rather than a CNN: chosen for a practical CPU-only baseline
  under time pressure. No CNN comparison has been conducted.
- Use a 9×9 local representation: chosen to preserve nearby spatial structure
  while keeping MLP input modest. Full-board and engineered-direction feature
  alternatives have not been tested.
- Mask framework-invalid actions: chosen to avoid wasting decisions on actions
  rejected by the environment. The learned model still decides among legal
  actions; no rule-based best-action policy is used.
- Train first in `coin-heaven`: this isolates navigation and visible coin
  collection before crates, bomb escape, and opponents are introduced.
- Keep experiments headless for training speed; use GUI only for qualitative
  inspection.
- Course-alignment check (17 September 2026): the instructor's curriculum
  starts with revealed-coin navigation (Task 1), then crates/bomb escape,
  opponents, and strong rule-based opponents. E01–E07 address Task 1 only.
  The course also requires the team to develop at least two different learned
  models. E01–E07 are variants of one DDQN model and do not by themselves meet
  that team-level requirement.

## Known Limitations / Failure Modes

- **Observed:** after E01, GUI evaluation moved once, collected one coin, then
  waited in place until the 400-step cap.
- **Measured:** E02 averaged 17.81 movement events and 5.02 collected coins per
  400-step round; E04 increased movement to 24.11 but reduced coins to 4.28.
- **Observed during exploratory training:** random bomb selection often caused
  early self-death before optimization warm-up.
- **Implementation limitation:** training resumes network weights but not replay,
  optimizer, target-network, epsilon, or environment-step state.
- **Not tested:** crate destruction, bomb escape reliability, opponent play,
  repeated seeds, Docker compatibility, and performance versus rule-based agents.
- **Observed in E11:** the first Task-2 candidate is safe in the three evaluated
  seeds but overuses bombs (about 19.65 bombs/round) while collecting few coins.
- **Observed in E11 GUI review:** it placed a crate-clearing bomb, then entered
  a two-tile loop. The number of reviewed seeds and loop frequency were not
  recorded.

## Experiment History

### E00 — DDQN baseline implementation and engineering verification

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Can a CPU-only DDQN agent satisfy the
  framework callback interface and execute the full replay/update/checkpoint
  path without runtime errors?
- **Model / algorithm:** current DDQN baseline described above.
- **Scenario and opponents:** `empty` and `coin-heaven`; one `ddqn_agent`; no
  opponents for smoke tests.
- **State, reward, and hyperparameters:** current definitions above.
- **Training rounds / steps:** one controlled smoke run reached 1,211 stored
  transitions and 53 optimizer updates; exact number of rounds not retained.
- **Random seed(s):** not recorded.
- **Training duration:** not recorded.
- **Starting checkpoint:** none for the initial smoke test. Its temporary
  checkpoint was removed before E01.
- **Exact training/evaluation commands:** several smoke commands were run;
  complete command history was not retained. **Not report-quality.**
- **Evaluation protocol:** import, callback, feature-shape, replay-sampling,
  and optimizer-path checks; not a policy-performance evaluation.
- **Quantitative result:** replay batch shape `(3, 495)`; Q-network output
  shape `(3, 6)` on CPU; 53 updates after the controlled smoke run.
- **Observed failure modes:** exploratory bombs could cause early death.
- **Interpretation:** measured engineering evidence supports that the pipeline
  executes; it does not establish policy quality.
- **Decision:** retain implementation and perform E01 training.
- **Files:** temporary smoke checkpoint removed; no result file retained.

### E01 — Initial coin-heaven training

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Can the initial DDQN learn coin collection
  in `coin-heaven` with no opponents or crates?
- **Model / algorithm:** current DDQN baseline.
- **Scenario and opponents:** `coin-heaven`; one `ddqn_agent`; no opponents.
- **State, reward, and hyperparameters:** current definitions above.
- **Training rounds / steps:** 1,000 rounds; 13,464 stored transitions; 3,117
  optimizer updates. Final epsilon 0.872; final logged Huber loss 3.5884.
- **Random seed(s):** 42.
- **Training duration:** not recorded.
- **Starting checkpoint:** none; the smoke-test checkpoint was removed first.
- **Exact training command:**

  ```powershell
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario coin-heaven --n-rounds 1000 --seed 42
  ```

- **Exact evaluation command:** E02 was the associated evaluation.
- **Evaluation protocol / quantitative result:** see E02.
- **Relevant observation:** training completed and checkpoint was written.
- **Failure modes:** not directly measured during this run; later GUI inspection
  found the wait-heavy behavior reported in E02.
- **Interpretation:** optimizer updates prove training occurred; loss is an
  optimization diagnostic, not evidence of performance.
- **Decision:** retain checkpoint for evaluation; do not claim it is sufficient.
- **Files:** `agent_code/ddqn_agent/ddqn_model.pt`; detailed E01 log was later
  overwritten by framework logging, so the completion values above are the
  contemporaneously recorded run summary.

### E02 — DDQN coin-heaven evaluation

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Does the E01 checkpoint collect coins in
  unseen `coin-heaven` rounds without exploration?
- **Model / algorithm:** E01 DDQN checkpoint, greedy action selection.
- **Scenario and opponents:** `coin-heaven`; one `ddqn_agent`; no opponents.
- **State, reward, and hyperparameters:** E01 configuration; reward is inactive
  during greedy evaluation.
- **Training rounds / steps:** not applicable; evaluation had 100 rounds and
  40,000 total environment steps.
- **Random seed(s):** 42.
- **Training duration:** not applicable. Evaluation wall-clock duration: not
  recorded. Callback time total: 25.4773 seconds.
- **Starting checkpoint:** `agent_code/ddqn_agent/ddqn_model.pt` from E01.
- **Exact training command:** not applicable; checkpoint came from E01.
- **Exact evaluation command:**

  ```powershell
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --scenario coin-heaven --n-rounds 100 --seed 42 --save-stats results\ddqn_coin_heaven_eval_100.json
  ```

- **Evaluation protocol:** 100 greedy, headless one-agent games with a fixed
  seed; one seed only.
- **Quantitative results:** 502 coins/score total; 5.02 coins/round; 1,781
  movement events (17.81/round); 0 suicides; all rounds reached 400 steps;
  callback time ≈0.000637 seconds/action.
- **Relevant GUI observation:** one displayed round moved once, collected one
  coin, then waited until the round cap. The displayed “win” meant the sole
  agent survived; it did not indicate strong coin collection.
- **Failure mode:** safe but inactive waiting policy.
- **Measured evidence vs interpretation:** inactivity is measured. Possible
  causes (weak wait penalty, sparse feedback, limited training, or local view)
  are interpretations and remain untested.
- **Decision:** do not advance curriculum yet; first test a targeted change to
  reduce inactivity using the same evaluation protocol.
- **Files:** `results/ddqn_coin_heaven_eval_100.json`, current checkpoint.

### E03 — Random-agent coin-heaven baseline

- **Date:** 17 September 2026.
- **Research question / hypothesis:** How does E02 compare with the supplied
  random baseline under the same scenario, seed, and round count?
- **Model / algorithm:** provided `random_agent`; no learning.
- **Scenario and opponents:** `coin-heaven`; one `random_agent`; no opponents.
- **State, reward, and hyperparameters:** not applicable / not recorded for the
  provided baseline.
- **Training rounds / steps:** no training; 100 evaluation rounds and 2,204
  total steps.
- **Random seed(s):** 42.
- **Training duration / starting checkpoint / exact training command:** not
  applicable.
- **Exact evaluation command:**

  ```powershell
  .\.venv\Scripts\python.exe main.py play --no-gui --agents random_agent --scenario coin-heaven --n-rounds 100 --seed 42 --save-stats results\random_coin_heaven_eval_100.json
  ```

- **Evaluation protocol:** 100 headless games; same seed and one-agent setup as
  E02.
- **Quantitative results:** 176 coins/score total; 1.76 coins/round; 100
  suicides; 947 invalid actions; 22.04 mean steps/round.
- **Relevant observations:** not recorded.
- **Failure mode:** random bombs lead to self-destruction in every round.
- **Interpretation:** on this single-seed comparison, E02 achieved more coins
  and fewer suicides. It is not a multi-seed statistical comparison.
- **Decision:** retain random results as a preliminary baseline; add repeated
  seeds before making stronger claims.
- **Files:** `results/random_coin_heaven_eval_100.json`.

### E04 — Coin-progress reward shaping (implemented; training not yet run)

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Can denser feedback for reducing shortest
  path distance to a visible coin, combined with a stronger wait penalty,
  reduce the inactive policy observed in E02?
- **Model / algorithm:** unchanged DDQN architecture from E01.
- **Scenario and opponents:** planned `coin-heaven`; one `ddqn_agent`; no
  opponents.
- **State representation:** unchanged 495-value representation.
- **Reward function and custom shaping:** change `WAITED` from -0.02 to -0.10.
  Add bounded shortest-path distance shaping: +0.10 when distance to the
  nearest reachable coin decreases by one; -0.10 when it increases by one;
  0 when unchanged. Coin-collection transitions receive their +5 event reward
  without a distance term.
- **Important hyperparameters:** unchanged from E01.
- **Training rounds / steps:** 1,000 training rounds; 12,906 stored transitions;
  2,977 optimizer updates. Final epsilon 0.877; final logged Huber loss 3.0399.
- **Random seed(s):** 42.
- **Training duration:** not recorded.
- **Starting checkpoint:** fresh initialization. E01 is preserved separately as
  `agent_code/ddqn_agent/ddqn_e01_coin_heaven.pt`.
- **Exact training command:**

  ```powershell
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario coin-heaven --n-rounds 1000 --seed 42
  ```
- **Exact evaluation command:**

  ```powershell
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --scenario coin-heaven --n-rounds 100 --seed 42 --save-stats results\ddqn_e04_coin_progress_eval_100.json
  ```

- **Evaluation protocol:** 100 greedy, headless one-agent games, seed 42; same
  protocol as E02.
- **Quantitative results:** 428 coins/score total; 4.28 coins/round; 2,411
  movement events (24.11/round); 0 suicides; all rounds reached 400 steps;
  callback time ≈0.000665 seconds/action.
- **Relevant observations / failure modes:** no GUI observation recorded. The
  policy remained far below the desired active coin-collection behavior.
- **Measured evidence vs interpretation:** unit tests verified the intended
  reward calculations. Compared with E02, E04 had 0.74 fewer coins/round and
  6.30 more moves/round on this seed. This establishes the observed trade-off
  for this run; it does not establish a general causal conclusion.
- **Decision:** reject E04 as the current reward configuration because it
  reduced the primary coin metric in the matched evaluation. Preserve its
  checkpoint and result file; next establish multi-seed comparisons before
  selecting a replacement change. The active code was restored to the E01
  reward configuration after recording this result.
- **Checkpoint / result filenames:**
  `agent_code/ddqn_agent/ddqn_e04_coin_progress.pt` and
  `results/ddqn_e04_coin_progress_eval_100.json`.

### E05 — Multi-seed comparison of E01, E04, and random baseline

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Does the E01-versus-E04 ordering observed
  on seed 42 persist across additional seeds?
- **Model / algorithm:** greedy E01 DDQN checkpoint, greedy E04 DDQN checkpoint,
  and the supplied `random_agent` baseline.
- **Scenario and opponents:** `coin-heaven`; one agent; no opponents.
- **State representation, reward, hyperparameters:** E01 uses the E01 settings;
  E04 uses the E04 reward variant; random-agent internals are not recorded.
- **Training rounds / steps:** no new training; each policy was evaluated for
  100 rounds on each of three seeds (300 rounds/policy).
- **Random seeds:** 42, 43, and 44.
- **Training duration / starting checkpoint:** not applicable. Evaluation wall
  clock duration: not recorded.
- **Exact training command:** not applicable.
- **Exact evaluation commands:** E01 used the E02 command with seeds 43 and 44
  and output files `results\e01_seed43.json` and `results\e01_seed44.json`.
  E04 used the same command with `DDQN_MODEL_FILE=ddqn_e04_coin_progress.pt`
  and output files `results\e04_seed43.json` and `results\e04_seed44.json`.
  The random baseline used the E03 command with seeds 43 and 44 and output
  files `results\random_seed43.json` and `results\random_seed44.json`.
- **Evaluation protocol:** 100 greedy/headless games per seed and policy; seed
  42 data comes from E02–E04, and seeds 43–44 were newly run.
- **Quantitative results (mean ± sample standard deviation, n=3 seeds):**

  | Policy | Coins/round | Moves/round | Suicides/round | Seconds/action |
  |---|---:|---:|---:|---:|
  | E01 | **4.71 ± 0.27** | 18.30 ± 1.83 | 0.00 ± 0.00 | 0.000614 ± 0.000020 |
  | E04 | 4.47 ± 0.24 | 23.10 ± 5.65 | 0.00 ± 0.00 | 0.000624 ± 0.000036 |
  | Random | 1.83 ± 0.18 | 11.72 ± 1.17 | 1.00 ± 0.00 | 0.000146 ± 0.000005 |

- **Relevant observations:** E04 was more active but remained lower in coin
  collection on average. No additional GUI observation was recorded.
- **Failure modes:** both DDQN variants remain far below the intended active
  coin-collection behavior despite surviving all evaluated rounds.
- **Measured evidence vs interpretation:** the three-seed mean favors E01 by
  0.24 coins/round. No formal hypothesis test was performed; this is evidence
  for configuration selection, not a claim of statistical significance.
- **Decision:** retain E01 as the active baseline and reject E04 reward shaping
  for the current model. Next investigate the state representation rather than
  applying another reward-only change.
- **Files:** all six seed-43/44 files listed above, plus E02/E03/E04 seed-42
  result files and both preserved checkpoints.

### E06 — Global nearest-coin state features

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Does supplying global direction and
  shortest-path distance to the nearest reachable coin improve collection over
  the local-view-only E01 representation?
- **Model / algorithm:** DDQN with unchanged hidden layers and E01 reward.
- **Scenario and opponents:** planned `coin-heaven`; one `ddqn_agent`; no
  opponents.
- **State representation:** E01's 495 values plus four BFS-derived values:
  normalized x direction, normalized y direction, normalized shortest-path
  distance, and a reachable-coin flag. New total: 499 values.
- **Reward / important hyperparameters:** E01 reward and all E01 DDQN
  hyperparameters unchanged.
- **Training rounds / steps:** 1,000 rounds; 13,684 stored transitions; 3,172
  optimizer updates. Final epsilon 0.870; final logged Huber loss 2.5154.
- **Random seeds:** training seed 42; evaluation seeds 42, 43, and 44.
- **Training duration:** not recorded.
- **Starting checkpoint:** fresh initialization because 495-input E01 weights
  are incompatible with 499-input E06 weights.
- **Exact training command:**

  ```powershell
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario coin-heaven --n-rounds 1000 --seed 42
  ```

- **Exact evaluation command:** the E02 command with E06 checkpoint active,
  repeated for seeds 42, 43, and 44; files are listed below.
- **Pre-training verification:** synthetic feature test returned shape `(499,)`
  and a coin two tiles to the right returned navigation values `(0.125, 0.0,
  0.0625, 1.0)`.
- **Evaluation protocol:** 100 greedy, headless one-agent games per seed; 300
  total evaluation rounds.
- **Quantitative results:** 4.58, 5.25, and 5.08 coins/round for seeds 42, 43,
  and 44 respectively: **4.97 ± 0.35** coins/round (sample standard deviation,
  n=3). Movement: **20.05 ± 2.95** moves/round. Mean callback time:
  **0.000649 ± 0.000008 seconds/action**. No suicides were reported.
- **Comparison with E01:** E01 achieved **4.71 ± 0.27** coins/round under the
  same three-seed protocol. E06's observed mean gain is 0.26 coins/round.
- **Relevant GUI observation:** on one seed-42 round, the agent moved roughly
  10–15 steps, collected multiple coins, and appeared to move toward coins
  without pausing between moves. It then stopped around step 40 and waited for
  the remainder of the round. It did not place a bomb.
- **Failure modes:** the agent remains relatively inactive for a 400-step round
  despite the improved mean score.
- **Measured evidence vs interpretation:** the three-seed sample mean favors
  E06. No formal hypothesis test was performed, so the evidence is sufficient
  to retain E06 as the current candidate but not to claim a definitive result.
- **Decision:** retain E06 as the active model and inspect it in the GUI before
  selecting the next training-budget or curriculum experiment.
- **Checkpoint / result filenames:**
  `agent_code/ddqn_agent/ddqn_e06_global_coin.pt`,
  `results/ddqn_e06_global_coin_eval_100.json`, `results/e06_seed43.json`, and
  `results/e06_seed44.json`.

### E07 — Longer E06 training budget

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Does increasing the training budget, while
  keeping E06 architecture, features, rewards, and seed fixed, reduce the
  early waiting behavior observed in the E06 GUI round?
- **Model / algorithm:** E06 DDQN architecture with 499 input features.
- **Scenario and opponents:** planned `coin-heaven`; one `ddqn_agent`; no
  opponents.
- **State representation, reward, hyperparameters:** unchanged from E06/E01.
- **Training rounds / steps:** planned 5,000 rounds but stopped after 3,183
  completed rounds once it had exceeded the 100,000-transition epsilon-decay
  schedule and remaining rounds became disproportionately slow. Actual totals:
  111,518 transitions and 27,630 optimizer updates. Final epsilon 0.05; final
  logged Huber loss 1.0775.
- **Random seed:** 42 for training; seeds 42–44 planned for evaluation.
- **Training duration:** not recorded.
- **Starting checkpoint:** fresh initialization; this isolates training budget
  as the experimental change. E06 checkpoint remains preserved.
- **Exact training command:**

  ```powershell
  $env:DDQN_MODEL_FILE = "ddqn_e07_long_train.pt"
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario coin-heaven --n-rounds 5000 --seed 42
  ```

- **Exact evaluation command (seed 42):**

  ```powershell
  $env:DDQN_MODEL_FILE = "ddqn_e07_long_train.pt"
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --scenario coin-heaven --n-rounds 100 --seed 42 --save-stats results\ddqn_e07_long_train_eval_100.json
  ```

- **Evaluation protocol:** 100 greedy, headless one-agent games for each of
  seeds 42, 43, and 44 (300 total evaluation rounds).
- **Quantitative results:**

  | Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD |
  |---|---:|---:|---:|---:|
  | Coins/round | 11.03 | 11.34 | 11.10 | **11.16 ± 0.16** |
  | Moves/round | 374.88 | 348.21 | 363.71 | 362.27 ± 13.39 |
  | Bombs/round | 5.47 | 8.02 | 5.18 | 6.22 ± 1.56 |
  | Suicides/round | 0.00 | 0.00 | 0.00 | 0.00 ± 0.00 |
  | Seconds/action | 0.000673 | 0.000653 | 0.000648 | 0.000658 ± 0.000014 |

- **Comparison with E06:** E06 achieved 4.97 ± 0.35 coins/round under the
  same three-seed protocol. E07's observed mean gain is 6.19 coins/round.
- **Relevant GUI observation:** the agent collected coins and used bombs. Across
  manually viewed seeds, it could eventually enter a two-tile movement loop
  near the end of a round. This is qualitative evidence only; loop frequency
  was not recorded.
- **Failure modes:** no suicides observed in `coin-heaven`; performance with
  crates, bomb escape, or opponents remains untested.
- **Measured evidence vs interpretation:** E07 consistently exceeds E06 on all
  three evaluated seeds. No formal hypothesis test was performed, but the
  magnitude and consistency support selecting E07 for the next curriculum step.
- **Decision:** select E07 as the active default checkpoint. Perform one GUI
  review, then begin a controlled `loot-crate`/bomb-survival experiment.
- **Checkpoint / result filenames:** `agent_code/ddqn_agent/ddqn_e07_long_train.pt`,
  `results/ddqn_e07_long_train_eval_100.json`, `results/e07_seed43.json`, and
  `results/e07_seed44.json`.

### E08 — Zero-shot loot-crate baseline

- **Date:** 17 September 2026.
- **Research question / hypothesis:** How does the selected Task-1 E07 policy
  perform on crates and hidden coins without any Task-2 training?
- **Model / algorithm:** E07 DDQN checkpoint, greedy evaluation.
- **Scenario and opponents:** `loot-crate`; one `ddqn_agent`; no opponents.
- **State representation / reward / hyperparameters:** E07's 499 features;
  reward inactive during evaluation; no new hyperparameters.
- **Training rounds / starting checkpoint:** no new training; E07 checkpoint.
- **Seeds / protocol:** 100 headless rounds each for seeds 42, 43, 44; 300
  evaluation rounds total.
- **Exact evaluation command:**

  ```powershell
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --scenario loot-crate --n-rounds 100 --seed <42|43|44> --save-stats results\e08_loot_crate_seed<seed>.json
  ```

- **Quantitative results (mean ± sample SD, n=3 seeds):** 0.00 ± 0.00
  coins/round; 0.28 ± 0.06 crates destroyed/round; 0.73 ± 0.25 bombs/round;
  0.073 ± 0.025 suicides/round; 190.49 ± 15.57 moves/round.
- **Observations / failure modes:** zero-shot Task-1 behavior can place bombs
  and destroy crates, but does not reveal/collect coins effectively and is not
  reliably safe around bombs.
- **Interpretation:** the result establishes a Task-2 baseline, not a failure
  of E07's Task-1 objective. It motivates scenario-specific training.
- **Decision:** proceed to E09 transfer fine-tuning in `loot-crate`.
- **Files:** `results/e08_loot_crate_seed42.json`,
  `results/e08_loot_crate_seed43.json`, and
  `results/e08_loot_crate_seed44.json`.

### E09 — Loot-crate transfer fine-tuning

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Can E07's learned navigation/bomb policy
  be adapted to crate destruction and hidden coins through Task-2 training?
- **Model / algorithm:** same 499-feature DDQN; initialize with a byte-identical
  copy of E07 weights, then train in `loot-crate`.
- **Scenario and opponents:** `loot-crate`; one `ddqn_agent`; no opponents.
- **State representation, reward, hyperparameters:** unchanged from E07.
- **Training budget / seed:** 1,000 rounds, seed 42; 9,473 transitions and
  2,119 optimizer updates; final epsilon 0.910; final logged loss 6.6550.
- **Starting checkpoint:** E07 weights, copied to a distinct E09 checkpoint to
  preserve E07 unchanged.
- **Exact training command:**

  ```powershell
  $env:DDQN_MODEL_FILE = "ddqn_e09_loot_finetune.pt"
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario loot-crate --n-rounds 1000 --seed 42
  ```

- **Evaluation protocol:** 100 greedy headless rounds for each of seeds 42, 43,
  and 44.
- **Quantitative results:** zero coins, bombs, crates, and suicides across all
  three evaluations. Movement was 156.44, 144.50, and 120.63 moves/round for
  seeds 42–44 respectively (140.52 ± 18.22 mean ± sample SD).
- **Failure mode:** fine-tuning collapsed to movement without bomb placement,
  so it could neither destroy crates nor reveal coins.
- **Interpretation:** this is consistent with the initially sparse/delayed
  crate reward, but causality is untested.
- **Decision:** reject E09 as the Task-2 candidate; test dense safe-bomb
  shaping in E10.
- **Files:** `agent_code/ddqn_agent/ddqn_e09_loot_finetune.pt`,
  `results/e09_loot_crate_seed42.json`, `results/e09_loot_crate_seed43.json`,
  and `results/e09_loot_crate_seed44.json`.

### E10 — Safe crate-bomb reward shaping

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Does rewarding a bomb only when it is
  adjacent to a crate and has an available escape path produce crate-clearing
  behavior without encouraging unsafe bomb spam?
- **Model / algorithm:** E07 499-feature DDQN initialized from preserved E07
  weights; no architecture change.
- **Scenario and opponents:** planned `loot-crate`; one agent; no opponents.
- **Reward change:** planned custom reward for a safe, crate-adjacent bomb and
  penalty for a bomb that is unsafe or cannot immediately affect a crate.
- **Other settings:** E09 training budget/seed and E07 hyperparameters retained.
- **Training:** 1,000 rounds, seed 42; 9,602 transitions and 2,151 updates;
  final epsilon 0.909; final logged loss 9.0488.
- **Exact training command:** E09 command with
  `DDQN_MODEL_FILE=ddqn_e10_safe_bomb.pt`.
- **Evaluation:** 100 greedy headless rounds, seed 42. Result: zero bombs,
  crates, coins, and suicides; 136.51 moves/round.
- **Failure mode / interpretation:** the custom reward is only observed after
  exploratory bomb placement, and the initial 1,000-round run retained very
  high exploration. This single-seed result is insufficient to establish the
  shaping rule's value, but insufficient behavior is clear.
- **Decision:** do not extend E10 unchanged. E11 retains this reward but changes
  training budget and epsilon-decay schedule.
- **Files:** `agent_code/ddqn_agent/ddqn_e10_safe_bomb.pt` and
  `results/e10_loot_crate_seed42.json`.

### E11 — Longer Task-2 training with faster epsilon decay

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Can E10's safe-crate-bomb shaping become
  effective when exploration decays on the scale of available Task-2
  transitions rather than E07's longer coin-heaven schedule?
- **Model / algorithm:** 499-feature DDQN initialized from E07 weights.
- **Scenario and opponents:** planned `loot-crate`; one agent; no opponents.
- **Changes from E10:** train 3,000 rather than 1,000 rounds; epsilon decays
  from 1.00 to 0.05 over 30,000 rather than 100,000 transitions. Safe-bomb
  shaping remains active. Other hyperparameters remain unchanged.
- **Seed / starting checkpoint:** seed 42; fresh copy of E07 weights.
- **Actual training budget:** planned 3,000 rounds; stopped after 1,659 completed
  rounds when later full-survival episodes made the remaining rounds
  disproportionately expensive. Actual totals: 48,128 transitions and 11,783
  optimizer updates; final epsilon 0.05; final logged Huber loss 0.6478.
- **Exact training command:**

  ```powershell
  $env:DDQN_MODEL_FILE = "ddqn_e11_loot_long_fast_eps.pt"
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario loot-crate --n-rounds 3000 --seed 42
  ```

- **Evaluation protocol:** 100 greedy headless rounds for each of seeds 42, 43,
  and 44 (300 total rounds).
- **Quantitative results (mean ± sample SD, n=3 seeds):** 0.070 ± 0.017
  coins/round; 1.597 ± 0.177 crates/round; 19.653 ± 0.622 bombs/round;
  0.000 ± 0.000 suicides/round; 229.847 ± 11.658 moves/round; and
  0.000569 ± 0.000005 seconds/action.
- **Measured comparison:** E08 zero-shot destroyed 0.283 ± 0.060 crates/round
  and collected zero coins; E09/E10 chose no bombs, crates, or coins in their
  evaluated seed(s). E11 is the first positive Task-2 result.
- **Failure mode:** bomb use is very frequent relative to crate destruction,
  and hidden-coin collection remains low.
- **Interpretation:** shorter epsilon decay plus safe-bomb shaping correlates
  with the improvement, but E11 changes both factors from E09, so their
  individual causal contributions are not isolated.
- **Decision:** retain E11 as the Task-2 candidate for qualitative GUI review;
  do not select it as the final agent yet. The next experiment should target
  bomb efficiency and coin collection rather than raw bomb frequency.
- **Files:** `agent_code/ddqn_agent/ddqn_e11_loot_long_fast_eps.pt`,
  `results/e11_loot_crate_seed42.json`, `results/e11_loot_crate_seed43.json`,
  and `results/e11_loot_crate_seed44.json`.

### E12 â€” Immediate-reversal penalty

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Can a small penalty for a direct,
  non-stationary reversal reduce the repeatedly observed two-tile loop while
  retaining E11's crate-clearing behavior?
- **Model / algorithm:** E11-compatible 499-feature DDQN, initialized from E11.
- **Scenario and opponents:** `loot-crate`; one agent; no opponents.
- **Change from E11:** add -0.20 reward when consecutive positions form
  `A → B → A`; all other active E11 settings remain unchanged.
- **Training budget / seed:** 750 rounds, seed 42; 7,759 stored transitions and
  1,690 optimizer updates. Final epsilon 0.754; final logged Huber loss 7.1433.
- **Exact training command:**

  ```powershell
  $env:DDQN_MODEL_FILE = "ddqn_e12_reversal_penalty.pt"
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario loot-crate --n-rounds 750 --seed 42
  ```

- **Evaluation protocol:** 100 greedy headless rounds for each of seeds 42, 43,
  and 44 (300 total rounds).
- **Quantitative results (mean ± sample SD, n=3 seeds):** 0.127 ± 0.025
  coins/round; 2.573 ± 0.227 crates/round; 17.503 ± 1.841 bombs/round;
  0.010 ± 0.010 suicides/round; 114.557 ± 15.236 moves/round; mean callback
  time 0.000585 seconds/action.
- **Measured comparison with E11:** crate destruction and coins increased by
  0.976 and 0.057 per round respectively, and bombs decreased by 2.150 per
  round. E12 introduced 0.010 suicides/round; E11 had none.
- **Failure modes:** direct loop frequency was not instrumented, so the
  experiment cannot establish whether the qualitative two-tile loop was removed.
- **Decision:** retain E12 as the current Task-2 checkpoint because it improves
  the primary Task-2 outcomes. Next, add a crate-navigation feature rather than
  increasing the reversal penalty.
- **Files:** `agent_code/ddqn_agent/ddqn_e12_reversal_penalty.pt`,
  `results/e12_loot_crate_seed42.json`, `results/e12_loot_crate_seed43.json`,
  and `results/e12_loot_crate_seed44.json`.

### E13 â€” Global crate-adjacent navigation feature

- **Date:** 17 September 2026.
- **Research question / hypothesis:** When coins are hidden, does a BFS-derived
  direction and distance to the nearest reachable free tile adjacent to a crate
  improve crate clearing and subsequent coin collection?
- **Model / algorithm:** DDQN with a new 503-feature input; fresh
  initialization is required because E12's 499-input first layer is incompatible.
- **Scenario and opponents:** `loot-crate`; one agent; no opponents.
- **State change:** append four values: normalized x/y offset, normalized
  shortest-path distance, and an availability flag for the nearest free tile
  adjacent to a crate. It guides navigation but not bomb selection.
- **Reward / hyperparameters:** active E12 reward settings, including safe-bomb
  shaping and immediate-reversal penalty; 30,000-step epsilon decay.
- **Pre-training verification:** a synthetic 17×17 board produced feature shape
  `(503,)` and crate-navigation values `(0.0625, 0.0, 0.03125, 1.0)` for a
  crate two cells to the right of the agent.
- **Training budget / seed:** 1,000 rounds, seed 42; 10,608 stored transitions
  and 2,403 optimizer updates. Final epsilon 0.664; final logged Huber loss
  1.9917.
- **Exact training command:**

  ```powershell
  $env:DDQN_MODEL_FILE = "ddqn_e13_crate_navigation.pt"
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario loot-crate --n-rounds 1000 --seed 42
  ```

- **Evaluation protocol:** 100 greedy headless rounds for each of seeds 42, 43,
  and 44 (300 total rounds).
- **Quantitative results (mean ± sample SD, n=3 seeds):** 0.000 ± 0.000
  coins/round; 0.000 ± 0.000 crates/round; 0.000 ± 0.000 bombs/round;
  0.000 ± 0.000 suicides/round; 260.347 ± 15.960 moves/round; mean callback
  time 0.000576 seconds/action.
- **Failure mode:** the policy moved through full 400-step games but never
  placed a bomb, so it could not reveal hidden coins.
- **Interpretation:** the feature computation was verified, but this result
  rejects the tested fresh-initialization configuration. It does not prove that
  crate navigation is intrinsically unhelpful.
- **Decision:** reject E13. Restore the E12-compatible 499-feature state and
  test improvements from the working E12 checkpoint.
- **Files:** `agent_code/ddqn_agent/ddqn_e13_crate_navigation.pt`,
  `results/e13_loot_crate_seed42.json`, `results/e13_loot_crate_seed43.json`,
  and `results/e13_loot_crate_seed44.json`.

### E14 â€” Stronger unsafe/wasted-bomb penalty

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Can doubling the penalty for a bomb that
  is unsafe or cannot hit an adjacent crate reduce E12's remaining bomb excess
  without losing its improved crate/coin outcomes?
- **Model / algorithm:** E12-compatible 499-feature DDQN initialized from E12.
- **Scenario and opponents:** `loot-crate`; one agent; no opponents.
- **Change from E12:** unsafe/wasted-bomb reward changes from -0.50 to -1.00.
  Safe crate-adjacent bomb reward (+0.75) and immediate-reversal penalty
  (-0.20) remain active.
- **Training budget / seed:** 750 rounds, seed 42; 7,807 stored transitions and
  1,702 optimizer updates. Final epsilon 0.753; final logged Huber loss 6.0192.
- **Exact training command:**

  ```powershell
  $env:DDQN_MODEL_FILE = "ddqn_e14_strong_unsafe_bomb.pt"
  .\.venv\Scripts\python.exe main.py play --no-gui --agents ddqn_agent --train 1 --scenario loot-crate --n-rounds 750 --seed 42
  ```

- **Evaluation protocol:** 100 greedy headless rounds for each of seeds 42, 43,
  and 44 (300 total rounds).
- **Quantitative results (mean ± sample SD, n=3 seeds):** 0.040 ± 0.017
  coins/round; 2.160 ± 0.044 crates/round; 18.933 ± 1.021 bombs/round;
  0.010 ± 0.000 suicides/round; and 118.607 ± 5.976 moves/round.
- **Measured comparison with E12:** all three primary Task-2 values worsened:
  coins fell by 0.087/round, crates by 0.413/round, and bombs increased by
  1.430/round.
- **Decision:** reject E14 and restore the E12 unsafe/wasted-bomb penalty of
  -0.50. E12 remains the leading Task-2 checkpoint.
- **Files:** `agent_code/ddqn_agent/ddqn_e14_strong_unsafe_bomb.pt`,
  `results/e14_loot_crate_seed42.json`, `results/e14_loot_crate_seed43.json`,
  and `results/e14_loot_crate_seed44.json`.

### E15 — Higher crate-destruction reward

- **Date:** 17 September 2026.
- **Research question / hypothesis:** Does increasing direct credit for a
  destroyed crate improve E12's crate-clearing and hidden-coin discovery?
- **Model / algorithm:** E12-compatible 499-feature DDQN initialized from E12.
- **Scenario and opponents:** `loot-crate`; one agent; no opponents.
- **Change from E12:** `CRATE_DESTROYED` reward changes from +1.00 to +2.00.
  Safe-bomb shaping, the -0.50 unsafe/wasted-bomb penalty, and the immediate
  reversal penalty remain active.
- **Status:** training and evaluation in progress.

## Planned Experiments (not completed)

### E15 completion — Higher crate-destruction reward

- **Training:** 750 rounds, seed 42; 7,515 stored transitions and 1,629 updates;
  final epsilon 0.762 and loss 6.6988.
- **Evaluation:** 100 greedy headless rounds per seed (42, 43, 44).
- **Results (mean ± sample SD):** coins 0.097 ± 0.032; crates 2.383 ± 0.146;
  bombs 9.850 ± 1.782; suicides 0.013 ± 0.006; moves 141.610 ± 15.039.
- **Decision:** rejected: E15 reduced bombs but was below E12 for crates and
  coins. `CRATE_DESTROYED` restored from +2.00 to +1.00.
- **Files:** `ddqn_e15_higher_crate_reward.pt` and `results/e15_loot_crate_seed*.json`.

### Curriculum strategy (planned)

- A new `crate-school` scenario has 35% crate density and 25 coins. It will be
  used before `loot-crate`, then `classic`; evaluation remains on original
  scenarios only.

### E16 — Immediate-danger action-safety features

- **Date:** 17 September 2026.
- **Research question:** Can explicit immediate-danger information improve
  survival and bomb handling without imposing a rule-based policy?
- **State change:** append six learned-input indicators, one per action, that
  encode legality and whether its resulting tile avoids active explosions and
  bomb blasts due on the next step. New input size: 505.
- **Verification:** synthetic state returned `(505,)` features and six safe
  actions on an empty board.
- **Status:** ready for fresh curriculum training; prior 499-input checkpoints
  remain incompatible but preserved.

### E17 — Safety-feature curriculum training

- **Date:** 17 September 2026.
- **Model:** E16 505-feature DDQN trained from scratch on `crate-school`.
- **Training:** 1,500 rounds, seed 42; 68,228 transitions; 16,808 updates;
  final epsilon 0.05 and loss 0.1178.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44.
- **Results (mean ± sample SD):** coins 1.270 ± 0.269; crates 4.437 ± 0.631;
  bombs 11.590 ± 0.464; suicides 0.073 ± 0.029; moves 146.917 ± 6.878.
- **Decision:** curriculum behavior is positive but survival is insufficient.
  Transfer the checkpoint to `loot-crate` for the next curriculum stage.

### E18 — Safety-feature transfer to loot-crate

- **Date:** 17 September 2026.
- **Model:** E16 505-feature checkpoint transferred to `loot-crate`.
- **Training:** 1,000 rounds, seed 42; 10,904 transitions; 2,477 updates;
  final epsilon 0.655 and loss 3.0445.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44.
- **Results (mean ± sample SD):** coins 0.370 ± 0.036; crates 3.087 ± 0.102;
  bombs 12.963 ± 3.845; suicides 0.033 ± 0.025; moves 99.147 ± 22.639.
- **Measured comparison:** E18 exceeds E12 in coins (0.370 vs 0.127) and crates
  (3.087 vs 2.573), but has more suicides (0.033 vs 0.010).
- **Decision:** retain E18 as the leading Task-2 performance checkpoint pending
  qualitative GUI inspection and targeted survival improvement.
- **GUI observation:** across manually viewed `loot-crate` seeds 42–44, the
  agent sometimes entered a two-tile movement loop and sometimes stopped
  completely. It did not die from bombs in the viewed rounds. This is
  qualitative evidence; the frequency of each behavior was not recorded.

### E19 — Target and temporal-context curriculum branch

- **Date:** 17 September 2026.
- **Model:** fresh 515-feature DDQN: E18 safety features, global nearest
  crate-adjacent navigation, and a six-value previous-action encoding.
- **Rationale:** E18 GUI review showed loops/stalling. The target feature gives
  the learner a goal when coins are hidden; temporal context lets a return move
  be represented differently from an otherwise identical first visit.
- **Status:** `crate-school` training in progress; E18 remains unchanged.

### E19 completion — Target and temporal-context curriculum branch

- **Training:** 1,500 rounds, seed 42; 60,085 transitions; 14,772 updates;
  final epsilon 0.05 and loss 0.2872.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44.
- **Results (mean ± sample SD):** coins 0.487 ± 0.142; crates 1.730 ± 0.457;
  bombs 8.263 ± 2.213; suicides 0.050 ± 0.010; moves 58.610 ± 9.786.
- **Decision:** reject the combined change: it is below E16 on both coins and
  crates. Test crate navigation without temporal context next to isolate the
  cause.

### E20 — Crate navigation ablation

- **Status:** fresh 509-feature curriculum run in progress: E16 safety features
  plus crate-target features, excluding E19 previous-action context.

### E20 completion — Crate navigation ablation

- **Training:** 1,500 rounds, seed 42; 62,239 transitions; 15,310 updates;
  final epsilon 0.05 and loss 0.2377.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44.
- **Results (mean ± sample SD):** coins 1.770 ± 0.131; crates 4.973 ± 0.410;
  bombs 10.020 ± 2.253; suicides 0.100 ± 0.046; moves 89.100 ± 16.932.
- **Measured comparison:** E20 exceeds E16 in coins (1.770 vs 1.270) and
  crates (4.973 vs 4.437), but suicides also rise (0.100 vs 0.073).
- **Decision:** retain as the strongest curriculum policy and transfer it to
  `loot-crate`; target survival separately after the transfer result is known.

### E21 — E20 transfer to loot-crate

- **Training:** 1,000 rounds, seed 42; 10,394 transitions; 2,349 updates;
  final epsilon 0.671 and loss 2.6179.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44.
- **Results (mean ± sample SD):** coins 0.500 ± 0.114; crates 2.637 ± 0.058;
  bombs 22.080 ± 1.599; suicides 0.060 ± 0.017; moves 139.700 ± 4.884.
- **Decision:** retain E21 as a coin-collection artifact, but select E18 as the
  balanced Task-2 checkpoint: E18 has more crates (3.087), fewer bombs (12.963)
  and fewer suicides (0.033). Begin `classic` evaluation with E18.

1. Add explicit evaluation counts for `WAIT`, `BOMB`, invalid actions, and
   deaths before moving to `loot-crate`.
2. Test a state-representation change that provides global coin direction or
   distance, while retaining E01 reward settings and the same evaluation protocol.
3. Curriculum experiments in `loot-crate`, then `classic`, including fixed
   baseline opponents.

## Final Results Summary

### E22 — E21 zero-shot classic baseline

- **Date:** 17 September 2026.
- **Model:** E21 509-feature checkpoint; no `classic` training.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44; no opponents.
- **Results (mean ± sample SD):** coins 0.093 ± 0.047; crates 2.580 ± 0.046;
  bombs 17.593 ± 2.041; suicides 0.020 ± 0.017; moves 159.540 ± 15.147.
- **Interpretation:** zero-shot baseline only. `classic` has nine coins, unlike
  the 50-coin curriculum scenarios, so raw coin totals are not directly comparable.
- **Decision:** train a separate E21-initialized `classic` fine-tuning branch;
  preserve E21 unchanged.

### E23 — Classic fine-tuning

- **Date:** 17 September 2026.
- **Research question:** Can direct training on the tournament-distribution
  `classic` scenario improve E21's scarce-coin collection without changing its
  architecture or reward configuration?
- **Model:** 509-feature E21 checkpoint copied to a separate E23 checkpoint.
- **Scenario:** `classic`, no opponents, training seed 42.
- **Status:** training in progress.

### E23 completion — Classic fine-tuning

- **Training:** 1,500 rounds, seed 42; 21,590 transitions; 5,148 updates;
  final epsilon 0.316 and loss 2.3194.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44.
- **Results (mean ± sample SD):** coins 0.030 ± 0.010; crates 2.553 ± 0.140;
  bombs 7.410 ± 0.360; suicides 0.000 ± 0.000; moves 93.160 ± 8.261.
- **Decision:** reject E23 for coin collection: it is safer but below E22's
  0.093 coins/round. Preserve E23 as a safety artifact; retain E21 for the
  first passive-opponent baseline.

Not available. Final checkpoint selection, comparison with the team's other
learned model, tournament-style evaluation, and Docker verification are not
yet completed.

### E24 — E21 against passive opponents

- **Date:** 17 September 2026.
- **Model:** E21 509-feature checkpoint; no opponent training.
- **Scenario/opponents:** `classic`; `peaceful_agent` and `coin_collector_agent`.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44.
- **Results (mean ± sample SD):** coins 0.213 ± 0.021; crates 3.897 ± 0.266;
  bombs 15.917 ± 1.595; suicides 0.290 ± 0.072; score 0.513 ± 0.067.
- **Interpretation:** compared with E22's no-opponent classic baseline, crate
  clearing rises but suicides increase sharply (0.020 to 0.290). This is a
  zero-shot Task-3 baseline, not an opponent-trained result.
- **Decision:** train an E21-initialized branch against the same passive
  opponents before considering the rule-based opponent.

### E25 — Passive-opponent fine-tuning

- **Training:** 1,000 `classic` rounds against `peaceful_agent` and
  `coin_collector_agent`, seed 42; 10,886 transitions; 2,472 updates; final
  epsilon 0.655 and loss 2.4847.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44, same opponents.
- **Results (mean ± sample SD):** coins 0.233 ± 0.057; crates 3.113 ± 0.075;
  bombs 5.487 ± 0.444; suicides 0.163 ± 0.038; score 0.333 ± 0.142.
- **Measured comparison with E24:** suicides fell from 0.290 to 0.163 and
  bombs from 15.917 to 5.487; coins rose slightly, while crates and score fell.
- **Decision:** retain E25 as the safer Task-3 candidate; establish a
  rule-based-opponent baseline before deciding whether to train against it.

### E26 — E25 against rule-based agent

- **Model:** E25 509-feature checkpoint; no rule-based-opponent training.
- **Scenario/opponent:** `classic`; one `rule_based_agent`.
- **Evaluation:** 100 greedy rounds per seed, 42/43/44.
- **Results (mean ± sample SD):** coins 0.300 ± 0.050; crates 4.010 ± 0.609;
  bombs 6.220 ± 0.736; suicides 0.253 ± 0.071; score 0.417 ± 0.076.
- **Decision:** retain as Task-4 zero-shot baseline. Train a separate E25
  checkpoint against the rule-based agent, then compare survival and score.

### E27 — Rule-based-opponent fine-tuning

- **Date:** 17 September 2026.
- **Research question:** Does 1,000 rounds of training against the
  `rule_based_agent` improve E25's Task-4 score or survival over its E26
  zero-shot baseline?
- **Model:** 509-feature DDQN initialized from E25 and saved separately as
  `ddqn_e27_rule_training.pt`.
- **Scenario/opponent:** `classic`; one `rule_based_agent`; training seed 42.
- **Training:** 1,000 rounds; 10,242 stored transitions; 2,311 optimizer
  updates; final epsilon 0.676; final logged Huber loss 1.4768.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds).
- **Results (mean ± sample SD, n=3 seeds):** coins 0.137 ± 0.057; crates
  3.753 ± 0.223; bombs 9.293 ± 1.377; suicides 0.317 ± 0.071; moves
  171.543 ± 14.893; score 0.187 ± 0.015; mean callback time approximately
  0.000734 seconds/action.
- **Measured comparison with E26:** all main outcomes worsened: score fell
  from 0.417 to 0.187, coins from 0.300 to 0.137, and crates from 4.010 to
  3.753; bombs and suicides rose from 6.220 and 0.253 to 9.293 and 0.317.
- **Decision:** reject E27 as the Task-4 candidate. Retain E25 unchanged as
  the stronger available checkpoint for further targeted survival work.
- **Files:** `agent_code/ddqn_agent/ddqn_e27_rule_training.pt`,
  `results/e27_rule_seed42.json`, `results/e27_rule_seed43.json`, and
  `results/e27_rule_seed44.json`.

### E28 — Consecutive-wait penalty

- **Date:** 17 September 2026.
- **Research question:** Can penalizing the second and later consecutive
  `WAIT` actions prevent the observed post-bomb freeze in E25?
- **Model:** 509-feature DDQN initialized from E25 and saved separately as
  `ddqn_e28_consecutive_wait.pt`.
- **Change from E25:** add -0.20 training reward for each second or later
  consecutive `WAIT`; the normal -0.02 `WAITED` reward and E12 reversal
  penalty remain active.
- **Scenario/opponents:** `classic`; `peaceful_agent` and
  `coin_collector_agent`; training seed 42.
- **Training:** 1,000 rounds. Exact transition/update totals were not retained
  in the framework log.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds).
- **Results (mean ± sample SD, n=3 seeds):** coins 0.197 ± 0.081; crates
  4.013 ± 0.203; bombs 13.367 ± 1.138; suicides 0.327 ± 0.031; moves
  193.663 ± 5.331; score 0.413 ± 0.224; mean callback time 0.000858
  seconds/action.
- **Measured comparison with E25:** movement increased by 94.516 actions/round,
  consistent with less waiting, and crates increased from 3.113 to 4.013.
  However, bombs rose from 5.487 to 13.367 and suicides from 0.163 to 0.327;
  coins also fell from 0.233 to 0.197.
- **Decision:** reject E28 as a replacement for E25. Test a smaller penalty
  beginning only after three consecutive waits to retain anti-stalling pressure
  without inducing excessive risky movement.
- **Files:** `agent_code/ddqn_agent/ddqn_e28_consecutive_wait.pt`,
  `results/e28_passive_seed42.json`, `results/e28_passive_seed43.json`, and
  `results/e28_passive_seed44.json`.

### E29 — Mild delayed consecutive-wait penalty

- **Date:** 17 September 2026.
- **Research question:** Can a smaller anti-stall penalty, applied only from
  the third consecutive `WAIT`, increase activity without E28's excessive
  bomb use and suicides?
- **Model:** 509-feature DDQN initialized from E25 and saved separately as
  `ddqn_e29_mild_wait_penalty.pt`.
- **Change from E25:** -0.05 training reward for each third or later
  consecutive `WAIT`; all other reward settings unchanged.
- **Scenario/opponents:** `classic`; `peaceful_agent` and
  `coin_collector_agent`; training seed 42.
- **Training:** 1,000 rounds. Exact transition/update totals were not retained
  in the framework log.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds).
- **Results (mean ± sample SD, n=3 seeds):** coins 0.197 ± 0.057; crates
  3.537 ± 0.264; bombs 6.270 ± 0.750; suicides 0.217 ± 0.049; moves
  159.437 ± 14.349; score 0.313 ± 0.106; mean callback time approximately
  0.000836 seconds/action.
- **Measured comparison with E25:** movement increased by 60.290 actions/round
  and crates by 0.424/round. Bombs and suicides increased modestly (by 0.783
  and 0.054/round), while coins and score fell by 0.036 and 0.020/round.
- **Decision:** retain E29 as the anti-stalling candidate pending qualitative
  GUI review. E25 remains the selected Task-3 checkpoint until that review;
  the observed score difference does not justify replacing it yet.
- **Files:** `agent_code/ddqn_agent/ddqn_e29_mild_wait_penalty.pt`,
  `results/e29_passive_seed42.json`, `results/e29_passive_seed43.json`, and
  `results/e29_passive_seed44.json`.

### E30 — Low-exploration transfer

- **Date:** 18 September 2026.
- **Research question:** Does starting E25 fine-tuning with much less random
  exploration reduce the observed greedy-policy stalling without using a
  hand-coded inference override?
- **Model:** 509-feature DDQN initialized from E25 and saved separately as
  `ddqn_e30_low_epsilon_transfer.pt`.
- **Change from E25:** epsilon decays from 0.20 to 0.05 over 30,000
  transitions, rather than from 1.00 to 0.05. Consecutive-wait shaping is
  disabled so this isolates the exploration change.
- **Scenario/opponents:** `classic`; `peaceful_agent` and
  `coin_collector_agent`; training seed 42.
- **Training:** 1,000 rounds; 125,037 environment transitions and 31,010
  gradient updates. Final epsilon 0.050; final logged Huber loss 0.4110.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds).
- **Results (mean ± sample SD, n=3 seeds):** coins 2.830 ± 0.263; crates
  49.783 ± 2.986; bombs 21.003 ± 1.392; suicides 0.533 ± 0.042; moves
  167.047 ± 10.430; score 4.063 ± 0.657.
- **Loop/stall instrumentation:** E30 recorded 99 diagnosable rounds per seed.
  Its mean maximum consecutive-wait streaks were 33.444, 30.828, and 30.404;
  mean direct reversals were 97.667, 97.485, and 99.071; mean four-step
  revisits were 85.929, 85.141, and 87.737. For the available E25 seed-42
  diagnostic baseline, the corresponding values were 96.586, 307.889, and
  281.061. This is measured evidence that E30 substantially reduced the
  observed stall and cycle behaviours on seed 42.
- **Measured comparison with E25:** E30 is much more active and productive,
  but its 0.533 suicides/round is unacceptable compared with E25's 0.163.
  Its higher score does not justify selecting it while survival is this poor.
- **Decision:** retain E30 as evidence that lower initial exploration addresses
  the loop/stall failure mode. Do not replace E25. The next controlled branch
  keeps E30's low-exploration schedule and tests a stronger training penalty
  for self-destruction.
- **Files:** `agent_code/ddqn_agent/ddqn_e30_low_epsilon_transfer.pt`,
  `results/e30_classic_seed42.json`, `results/e30_classic_seed43.json`,
  `results/e30_classic_seed44.json`, and
  `results/e30_diagnostics_seed42.log` through
  `results/e30_diagnostics_seed44.log`.

### E31 — Self-preservation transfer

- **Date:** 18 September 2026.
- **Research question:** Can a stronger self-death consequence retain E30's
  measured reduction in stalling while reducing its unacceptable suicide rate?
- **Model:** 509-feature DDQN initialized from E30 and saved separately as
  `ddqn_e31_self_preservation.pt`.
- **Change from E30:** `KILLED_SELF` reward is -30.00 instead of -12.00.
  The E30 epsilon schedule (0.20 to 0.05 over 30,000 transitions), scenario,
  opponents, seed, and disabled consecutive-wait shaping are retained.
- **Training:** 1,000 rounds; 105,140 environment transitions and 26,036
  gradient updates. Final epsilon 0.050; final logged Huber loss 0.6462.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds).
- **Results (mean ± sample SD, n=3 seeds):** coins 3.007 ± 0.163; crates
  59.570 ± 1.461; bombs 23.097 ± 0.733; suicides 0.307 ± 0.025; moves
  186.210 ± 6.087; score 4.423 ± 0.352.
- **Loop/stall instrumentation:** mean maximum wait streaks were 63.667,
  68.364, and 68.869 across seeds 42–44; direct reversals were 122.899,
  133.788, and 133.354; four-step revisits were 109.202, 119.475, and
  119.525. This is worse than E30 but remains substantially lower than the
  available E25 seed-42 baseline (96.586, 307.889, and 281.061).
- **Measured comparison with E30:** suicides fell by 0.227/round, and coins,
  crates, moves, and score increased. Bomb use rose by 2.093/round and the
  remaining 0.307 suicides/round is still too high.
- **Decision:** retain E31 as the leading active-performance checkpoint, but
  do not replace the safer E25 submission/default checkpoint. Test a stronger
  unsafe/wasted-bomb penalty next while preserving E31's settings.
- **Files:** `agent_code/ddqn_agent/ddqn_e31_self_preservation.pt`,
  `results/e31_classic_seed42.json`, `results/e31_classic_seed43.json`,
  `results/e31_classic_seed44.json`, and
  `results/e31_diagnostics_seed42.log` through
  `results/e31_diagnostics_seed44.log`.

### E32 — Bomb-discipline transfer

- **Date:** 18 September 2026.
- **Research question:** Can discouraging unsafe or wasted bomb placements
  reduce E31's excessive bomb use and remaining suicides without returning to
  its former stall behaviour?
- **Model:** 509-feature DDQN initialized from E31 and saved separately as
  `ddqn_e32_bomb_discipline.pt`.
- **Change from E31:** unsafe/wasted-bomb shaping is -1.00 rather than -0.50.
  The +0.75 safe crate-adjacent-bomb reward, -30.00 self-death reward,
  low-exploration schedule, scenario, opponents, and seed are retained.
- **Training:** 1,000 rounds; 128,463 environment transitions and 31,866
  gradient updates. Final epsilon 0.050; final logged Huber loss 0.5346.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds).
- **Results (mean ± sample SD, n=3 seeds):** coins 3.053 ± 0.297; crates
  56.927 ± 2.632; bombs 18.790 ± 1.060; suicides 0.280 ± 0.036; moves
  181.363 ± 15.642; score 4.220 ± 0.487.
- **Loop/stall instrumentation:** mean maximum wait streaks were 86.566,
  65.909, and 70.273; direct reversals were 170.051, 150.192, and 164.566;
  four-step revisits were 156.434, 136.889, and 150.899 across seeds 42–44.
  These are higher than E31 but remain below the available E25 seed-42
  baseline (96.586, 307.889, and 281.061).
- **Measured comparison with E31:** bombs fell by 4.307/round and suicides by
  0.027/round; coins increased slightly. Crates and score declined slightly,
  and the remaining 0.280 suicides/round is still above the safety target.
- **Decision:** retain E32 as the current balanced active-performance
  checkpoint, while preserving E25 as the safer default/submission checkpoint.
  The next branch should supply prospective bomb-escape information to the
  learner, rather than continuing to increase reward penalties blindly.
- **Files:** `agent_code/ddqn_agent/ddqn_e32_bomb_discipline.pt`,
  `results/e32_classic_seed42.json`, `results/e32_classic_seed43.json`,
  `results/e32_classic_seed44.json`, and
  `results/e32_diagnostics_seed42.log` through
  `results/e32_diagnostics_seed44.log`.

### E33 — Prospective bomb-escape learned feature

- **Date:** 18 September 2026.
- **Research question:** Can explicitly supplying whether a bomb placed now can
  clear an adjacent crate and still reach a non-blast free tile reduce E32's
  remaining self-destruction without imposing a rule-based action choice?
- **Model:** 510-feature DDQN initialized by widening E32's first layer. The
  first 509 input weights are preserved; the new feature weight is initialized
  to zero, so the migrated network initially reproduces E32 up to normal
  floating-point arithmetic.
- **State change:** append one binary prospective-bomb-escape feature. It
  models the newly placed bomb at the agent's tile, checks for an adjacent
  crate, and searches four movement steps for a tile outside that bomb's blast.
  It is only provided to the MLP; it never masks, forces, or otherwise selects
  the `BOMB` action.
- **Compatibility:** this feature is opt-in through
  `DDQN_USE_PROSPECTIVE_BOMB_SAFETY=1`; the default representation and E25
  submission checkpoint remain 509-feature compatible.
- **Verification:** synthetic 17×17 state checks returned feature dimensions
  509 (default) and 510 (opt-in), with the appended feature equal to 1.0 for a
  crate-adjacent, escapable bomb. Migrated E33 Q-values with the appended input
  set to zero matched E32 within 3.1e-5.
- **Training configuration:** starts from E32; retains E32's -30.00 self-death
  reward, -1.00 unsafe/wasted-bomb reward, low-exploration schedule, scenario,
  opponents, and seed.
- **Training:** 1,000 rounds; 127,946 environment transitions and 31,737
  gradient updates. Final epsilon 0.050; final logged Huber loss 1.7464.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds) with `DDQN_USE_PROSPECTIVE_BOMB_SAFETY=1` set for every
  run.
- **Results (mean ± sample SD, n=3 seeds):** coins 3.073 ± 0.142; crates
  55.463 ± 1.352; bombs 17.507 ± 0.301; suicides 0.320 ± 0.020; moves
  161.430 ± 5.442; score 3.857 ± 0.417.
- **Loop/stall instrumentation:** mean maximum wait streaks were 71.505,
  75.828, and 91.323; direct reversals were 154.859, 160.273, and 161.596;
  four-step revisits were 140.960, 146.455, and 148.919 across seeds 42–44.
- **Measured comparison with E32:** bomb use fell by 1.283/round and coins
  increased slightly, but suicides rose from 0.280 to 0.320/round and score
  fell by 0.363/round. The feature is operational but did not improve the
  selected safety-performance trade-off in this transfer.
- **Decision:** reject E33 as a replacement for E32. Preserve the feature
  implementation and checkpoint for reproducibility; retain E32 as the active
  performance candidate and E25 as the safer default/submission checkpoint.
- **Files:** `agent_code/ddqn_agent/ddqn_e33_prospective_bomb_safety.pt`,
  `results/e33_classic_seed42.json`, `results/e33_classic_seed43.json`,
  `results/e33_classic_seed44.json`, and
  `results/e33_diagnostics_seed42.log` through
  `results/e33_diagnostics_seed44.log`.

### E34 — Extended E32 weight continuation

- **Date:** 18 September 2026.
- **Research question:** Does another matched training budget from the selected
  E32 weights improve the activity/survival trade-off without another reward
  or representation change?
- **Model:** 509-feature DDQN initialized from E32 and saved separately as
  `ddqn_e34_extended_e32.pt`.
- **Configuration:** retains E32's -30.00 self-death reward, -1.00
  unsafe/wasted-bomb reward, epsilon schedule (0.20 to 0.05 over 30,000
  transitions), disabled consecutive-wait shaping, scenario, opponents, and
  seed. The E33 prospective-bomb feature is disabled.
- **Important limitation:** this is a weight continuation, not a complete
  training-state resume: the current implementation creates fresh replay,
  optimizer, target-network, and epsilon-progress state for every training
  invocation.
- **Training:** 1,000 rounds; 133,809 environment transitions and 33,203
  gradient updates. Final epsilon 0.050; final logged Huber loss 0.6474.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds).
- **Results (mean ± sample SD, n=3 seeds):** coins 3.060 ± 0.182; crates
  58.543 ± 1.259; bombs 23.660 ± 0.372; suicides 0.250 ± 0.070; moves
  206.137 ± 13.812; score 3.993 ± 0.232.
- **Loop/stall instrumentation:** mean maximum wait streaks were 57.354,
  61.535, and 42.444; direct reversals were 142.394, 167.414, and 123.061;
  four-step revisits were 126.404, 148.990, and 105.758 across seeds 42–44.
- **Measured comparison with E32:** coins remained comparable, crates rose by
  1.616/round, suicides fell by 0.030/round, and all recorded stall metrics
  improved. Bombs rose by 4.870/round and score fell by 0.227/round.
- **Decision:** retain E34 alongside E32 as the current active-performance
  trade-off: E34 is more active and slightly safer; E32 uses fewer bombs and
  has the higher observed score. Neither replaces E25 as the safer default.
- **Files:** `agent_code/ddqn_agent/ddqn_e34_extended_e32.pt`,
  `results/e34_classic_seed42.json`, `results/e34_classic_seed43.json`,
  `results/e34_classic_seed44.json`, and
  `results/e34_diagnostics_seed42.log` through
  `results/e34_diagnostics_seed44.log`.

### E35 — Dueling-DDQN transfer

- **Date:** 18 September 2026.
- **Research question:** After reward, safety-feature, and training-budget
  branches reached a persistent activity/survival trade-off, can a dueling
  value/advantage decomposition improve action values without resetting the
  learned E34 policy?
- **Model:** 509-feature dueling DDQN. It has the same two hidden layers as
  the standard MLP, then separate scalar value and six-action advantage heads;
  Q-values are `V + A - mean(A)`.
- **Initialization:** E34's two hidden layers and final Q head are migrated.
  The old final Q head initializes the advantage head, while its action mean
  initializes the value head. This makes the initial dueling Q-values match
  E34 within 6.2e-5 on a synthetic batch, preserving action rankings before
  learning begins.
- **Compatibility:** checkpoints now record architecture metadata. Legacy
  checkpoints are interpreted as `standard`, so E25–E34 remain loadable under
  the default configuration. E35 requires `DDQN_ARCHITECTURE=dueling`.
- **Training:** 1,000 rounds; 120,245 environment transitions and 29,812
  gradient updates. Final epsilon 0.050; final logged Huber loss 0.7873.
- **Evaluation:** 100 greedy headless rounds for each of seeds 42, 43, and 44
  (300 total rounds), with `DDQN_ARCHITECTURE=dueling` enabled.
- **Results (mean ± sample SD, n=3 seeds):** coins 3.283 ± 0.112; crates
  58.640 ± 2.030; bombs 17.933 ± 0.626; suicides 0.190 ± 0.062; moves
  186.450 ± 5.217; score 4.383 ± 0.204.
- **Loop/stall instrumentation:** mean maximum wait streaks were 113.374,
  89.636, and 85.586; direct reversals were 206.919, 182.798, and 174.424;
  four-step revisits were 190.667, 167.687, and 159.121 across seeds 42–44.
  These are worse than E34 but remain below the available E25 seed-42
  reversal/revisit baseline (307.889 and 281.061).
- **Measured comparison with E34:** coins rose by 0.223/round, bombs fell by
  5.727/round, suicides fell by 0.060/round, and score rose by 0.390/round.
  E35 is less active and has higher loop metrics than E34, so qualitative GUI
  review and Task-4 testing remain necessary.
- **Decision:** retain E35 as the leading active-performance checkpoint. Keep
  E25 unchanged as the conservative default/submission fallback until E35 has
  passed zero-shot and trained evaluation against rule-based opponents.
- **Files:** `agent_code/ddqn_agent/ddqn_e35_dueling_transfer.pt`,
  `results/e35_classic_seed42.json`, `results/e35_classic_seed43.json`,
  `results/e35_classic_seed44.json`, and
  `results/e35_diagnostics_seed42.log` through
  `results/e35_diagnostics_seed44.log`.

### E36 — E35 zero-shot rule-based-opponent evaluation

- **Date:** 18 September 2026.
- **Research question:** Does the selected E35 dueling policy transfer to a
  strong `rule_based_agent` before receiving any opponent-specific training?
- **Model:** E35 509-feature dueling checkpoint; no new training.
- **Scenario/opponent:** `classic`; one `rule_based_agent`.
- **Evaluation:** 100 greedy headless rounds per seeds 42, 43, and 44 (300
  total rounds), with `DDQN_ARCHITECTURE=dueling`.
- **Results (mean ± sample SD, n=3 seeds):** coins 3.337 ± 0.081; crates
  58.767 ± 1.170; bombs 20.957 ± 0.860; suicides 0.370 ± 0.046; moves
  182.507 ± 10.594; score 3.553 ± 0.110.
- **Loop/stall instrumentation:** mean maximum wait streaks were 35.455,
  26.687, and 38.091; direct reversals were 97.747, 106.111, and 110.263;
  four-step revisits were 79.788, 86.242, and 89.869 across seeds 42–44.
- **Measured comparison with E26:** compared with E25's 0.417 score, 0.300
  coins, and 4.010 crates per round against the same opponent class, E36 is
  substantially more productive. Its suicides are higher (0.370 vs 0.253),
  so this is a strong zero-shot Task-4 baseline rather than a final selection.
- **Decision:** retain E35 unchanged. Test a separate low-exploration
  rule-based-opponent fine-tuning branch focused on reducing the remaining
  self-destruction without erasing E35's zero-shot strength.
- **Files:** `results/e36_e35_rule_seed42.json`,
  `results/e36_e35_rule_seed43.json`, `results/e36_e35_rule_seed44.json`, and
  `results/e36_diagnostics_seed42.log` through
  `results/e36_diagnostics_seed44.log`.

### E37 — Low-exploration rule-based-opponent fine-tuning

- **Date:** 18 September 2026.
- **Research question:** Can E35 adapt to `rule_based_agent` play while
  retaining its strong zero-shot policy if training exploration remains at the
  minimum epsilon rather than repeating E27's high-exploration transfer?
- **Model:** 509-feature dueling DDQN initialized from E35 and saved separately
  as `ddqn_e37_dueling_rule_finetune.pt`.
- **Configuration:** `epsilon=0.05` throughout training; retains E35's -30.00
  self-death reward, -1.00 unsafe/wasted-bomb reward, disabled consecutive-wait
  shaping, scenario, opponent, and seed 42.
- **Training:** 1,000 rounds; 146,885 environment transitions and 36,472
  gradient updates. Final epsilon 0.050; final logged Huber loss 0.5302.
- **Evaluation:** 100 greedy headless rounds per seeds 42, 43, and 44 (300
  total rounds), with `DDQN_ARCHITECTURE=dueling`.
- **Results (mean ± sample SD, n=3 seeds):** coins 2.603 ± 0.195; crates
  51.270 ± 1.572; bombs 18.137 ± 0.771; suicides 0.550 ± 0.020; moves
  166.633 ± 3.822; score 2.703 ± 0.064.
- **Loop/stall instrumentation:** mean maximum wait streaks were 43.828,
  38.414, and 40.737; direct reversals were 108.606, 96.253, and 116.162;
  four-step revisits were 90.374, 79.899, and 100.293 across seeds 42–44.
- **Measured comparison with E36:** bombs fell by 2.820/round but coins,
  crates, score, and moves all declined. Suicides increased from 0.370 to
  0.550/round. The lower epsilon did not preserve E35's zero-shot strength.
- **Decision:** reject E37. Retain E35 as the Task-4 candidate; do not apply
  rule-opponent fine-tuning to the selected checkpoint again without a new,
  independently motivated design change.
- **Files:** `agent_code/ddqn_agent/ddqn_e37_dueling_rule_finetune.pt`,
  `results/e37_rule_seed42.json`, `results/e37_rule_seed43.json`,
  `results/e37_rule_seed44.json`, and
  `results/e37_diagnostics_seed42.log` through
  `results/e37_diagnostics_seed44.log`.

### E38 — Tournament-style random-opponent evaluation

- **Date:** 18 September 2026.
- **Research question:** Does the selected E35 dueling policy run correctly and
  remain competitive in the submission-test configuration of one learned agent
  against three `random_agent` opponents?
- **Model:** E35 dueling checkpoint; no new training.
- **Scenario/opponents:** `classic`; three `random_agent` opponents.
- **Evaluation:** 100 greedy headless rounds per seeds 42, 43, and 44 (300
  total rounds), with `DDQN_ARCHITECTURE=dueling`.
- **Results (mean ± sample SD, n=3 seeds):** coins 3.267 ± 0.189; crates
  58.180 ± 2.138; bombs 15.073 ± 0.214; suicides 0.043 ± 0.023; moves
  245.223 ± 12.954; score 3.267 ± 0.189.
- **Loop/stall instrumentation:** mean maximum wait streaks were 105.444,
  112.343, and 111.707; direct reversals were 286.808, 285.081, and 269.202;
  four-step revisits were 279.232, 278.434, and 261.747 across seeds 42–44.
  These remain a qualitative/diagnostic weakness despite strong survival and
  productivity; random-opponent early deaths change the game dynamics relative
  to E35's weaker and rule-based opponent evaluations.
- **Decision:** select E35 as the packaged default checkpoint. It is the first
  candidate with strong measured performance across weaker opponents, a
  `rule_based_agent`, and the three-random-agent submission-style setup. E25
  remains preserved as the conservative standard-DDQN fallback.
- **Files:** `results/e38_e35_random_seed42.json`,
  `results/e38_e35_random_seed43.json`, `results/e38_e35_random_seed44.json`,
  and `results/e38_diagnostics_seed42.log` through
  `results/e38_diagnostics_seed44.log`.

### E39 — Short-cycle revisit penalty

- **Date:** 18–19 September 2026.
- **Research question:** Can a small penalty for returning to any of the prior
  four positions, when the transition has not collected a coin, destroyed a
  crate, or killed an opponent, reduce the observed movement loops without
  reducing E35's productive behavior?
- **Model:** 509-feature dueling DDQN initialized from E35 and saved as
  `ddqn_e39_short_cycle.pt`.
- **Change from E35:** enable a training-only `-0.05` short-cycle penalty.
  It is applied only to a non-stationary revisit among the most recent four
  positions with no objective event. The existing E12 immediate-reversal
  penalty remains active. Consecutive-wait shaping remains disabled.
- **Scenario/opponents:** `classic`; `peaceful_agent` and
  `coin_collector_agent`; training seed 42.
- **Training:** an initial background attempt stopped at round 163 without a
  Python error and its partial checkpoint was preserved. The supervised rerun
  completed 1,000 rounds: 166,285 transitions and 41,322 gradient updates;
  final epsilon 0.050 and final logged Huber loss 0.8437.
- **Evaluation:** 100 greedy headless rounds per seeds 42, 43, and 44 (300
  total rounds), with `DDQN_ARCHITECTURE=dueling`.
- **Results (mean ± sample SD, n=3 seeds):** coins 2.780 ± 0.242; crates
  52.577 ± 3.045; bombs 17.273 ± 1.001; suicides 0.193 ± 0.042; moves
  205.750 ± 4.701; score 4.047 ± 0.544; callback time 0.000629 ± 0.000033
  seconds/action.
- **Measured comparison with E35:** bombs fell by 0.660/round, but coins fell
  by 0.503/round, crates by 6.063/round, and score by 0.336/round. Suicides
  were effectively unchanged (0.193 versus 0.190/round). The added penalty
  therefore did not achieve the intended loop/survival trade-off.
- **Decision:** reject E39. The short-cycle penalty is disabled by default in
  code and E35 remains the selected packaged checkpoint.
- **Files:** `agent_code/ddqn_agent/ddqn_e39_short_cycle.pt`, preserved
  `ddqn_e39_short_cycle_attempt1_partial.pt`, and
  `results/e39_eval_seed42.json` through `results/e39_eval_seed44.json`.

### E40 — Four-step action-escape features

- **Date:** 19 September 2026.
- **Research question:** Can supplying the learned policy with an explicit
  four-step escape-route indicator for each action reduce E35's self-destruction
  while retaining its crate-clearing performance?
- **Model:** 515-feature dueling DDQN initialized from E35 and saved as
  `ddqn_e40_escape_features.pt`.
- **State change:** append six values, one per action. Each value is one only
  when a small time-expanded search finds a legal sequence of moves/waits that
  avoids active explosions and bomb blasts over the next four steps. For BOMB,
  the candidate's own four-step fuse is included. The values are inputs to the
  network only; they do not mask, force, or otherwise select actions.
- **Migration verification:** copied E35's 509 existing first-layer input
  columns and zero-initialized the six appended columns. On random test inputs
  with the new values set to zero, maximum E35/E40 Q-value difference was
  0.00000000. Synthetic tests returned `(515,)` features, six safe actions on
  an empty board, and no safe action in a deliberately trapped bomb corridor.
- **Scenario/opponents:** `classic`; `peaceful_agent` and
  `coin_collector_agent`; training seed 42. E35's low-epsilon, safety-weighted
  training configuration was retained.
- **Training:** 1,000 rounds; 168,176 transitions and 41,795 gradient updates;
  final epsilon 0.050 and final logged Huber loss 0.4158.
- **Evaluation:** 100 greedy headless rounds per seeds 42, 43, and 44 (300
  total rounds), with `DDQN_ARCHITECTURE=dueling` and
  `DDQN_USE_ACTION_ESCAPE_FEATURES=1`.
- **Results (mean ± sample SD, n=3 seeds):** coins 2.690 ± 0.157; crates
  52.183 ± 3.300; bombs 16.957 ± 1.176; suicides 0.157 ± 0.031; moves
  143.210 ± 7.781; score 3.673 ± 0.483; callback time 0.000771 ± 0.000046
  seconds/action.
- **Measured comparison with E35:** suicides fell by 0.033/round and bombs by
  0.976/round, but coins fell by 0.593/round, crates by 6.457/round, moves by
  43.240/round, and score by 0.710/round. The safety feature lowers risk but
  makes the learned policy materially less productive.
- **Decision:** reject E40 as the selected policy. Keep the implementation
  disabled by default and retain E35 as the packaged 509-feature default.
- **Files:** `agent_code/ddqn_agent/ddqn_e40_escape_features.pt`,
  `agent_code/ddqn_agent/migrate_e35_to_e40.py`, and
  `results/e40_eval_seed42.json` through `results/e40_eval_seed44.json`.

## EXTRAINFO

- `coin-heaven` contains no crates and has 50 generated coins, so it isolates
  navigation toward currently collectable coins.
- A Q-value estimates discounted future shaped reward. DDQN separates action
  selection (online network) from action evaluation (target network) to reduce
  overestimation relative to ordinary DQN.
