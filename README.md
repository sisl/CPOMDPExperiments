# CPOMDPExperiments

Using Julia 1.6

To install, in a Julia REPL:

1. Activate the local environment with `] activate .` 

2. Develop the constrained solver packages with `include("develop.jl")`

3. Install remaining dependencies from the Manifest using `] instantiate`

To run experiments in the installed environment, include an experiment file `julia --project=. [filepath]`. Experiment files are

    - `experiments/run_*.jl` for the main comparison
    - `experiments/pareto_frontier.jl` to generate the pareto plot
    - `experiments/cost_prop_exp.jl` to compare minimal cost propagation

Experiments result files are included in the `results` folder. Summaries can be quickly printed with the `load_and_print` function, e.g.:

```
julia --project=. -e 'using CPOMDPExperiments; load_and_print("results/spillpoint_pft_10sims.jld2")'
```

### Installation bugs

Runtime warnings are expected. If installation errors occur from trying to install the frozen manifest, you can try direct package installation:

1. Deleting the `Manifest.toml` file

2. From the `Project.toml` file, remove the lines for the four packages under development

3. Run `julia --project=. install.jl`

### Citation

If you found this repo useful, please cite our ICAPS paper (available on Arxiv [here](https://arxiv.org/abs/2212.12154)).

```
@inproceedings{jamgochian2023online,
  author = {Arec Jamgochian and Anthony Corso and Mykel J. Kochenderfer},
  title = {Online Planning for Constrained {POMDP}s with Continuous Spaces through Dual Ascent}
  booktitle = {International Conference on Automated Planning and Scheduling (ICAPS)},
  year = {2023}
}
```
