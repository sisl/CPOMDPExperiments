# CPOMDPExperiments

Using Julia 1.6

To install:

1. Remove dependencies in `Project.toml` file by deleting the lines under `[deps]`. 

2. From a Julia REPL run `include("install.jl")`


To run experiments in a Julia REPL:

1. Activate the local environment with `] activate .`

2. Include an experiment file `include(filepath)`. Experiment files are
    - `experiments/run_*.jl` for the main comparison
    - `experiments/pareto_frontier.jl` to generate the pareto plot
    - `experiments/cost_prop_exp.jl` to compare minimal cost propagation

Experiments result files are included in the `results` folder. Summaries can be quickly printed with the `load_and_print` function, e.g.:

```
julia -e 'import Pkg; Pkg.activate("."); using CPOMDPExperiments; load_and_print("results/spillpoint_pft_10sims.jld2")'
```