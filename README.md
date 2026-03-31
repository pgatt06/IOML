# Optimal Classification Trees Project

Final student submission for the IOML project on optimal classification trees.

## Project Scope

This project involves a core comparison using five datasets:
- `iris`
- `seeds`
- `wine`
- `banknote`
- `wdbc`

Additionally, two larger datasets, `titanic` and `diabetes`, are used for complementary experiments:
- Extended exact runs
- Open question 3 on feature augmentation

## Repository Structure

### Main Files
- **Report**: `rapport/rapport.pdf`
- **Report Source**: `rapport/rapport.tex`
- **Exact Formulation**: `src/main.jl`
- **Grouped Formulation (`F_U`)**: `src/main_merge.jl`
- **Iterative Formulations (`F_h^S` and `F_e^S`)**: `src/main_iterative_algorithm.jl`
- **Feature Augmentation Study (Open Question 3)**: `src/test_q3.jl`
- **Opposite-Corner Study (Open Question 1)**: `src/main_q4_opposite_corner.jl`

The scripts `src/main.jl`, `src/main_merge.jl`, and `src/main_iterative_algorithm.jl` are identical to the official versions provided in `/Users/p.gatt/Documents/3A/IOML/Project/src`.

### Results
- **Exact Formulation**: `Results/main/results.txt`
- **Grouped Formulation**: `Results/main_merge/results.txt`
- **Iterative Formulations**: `Results/main_iterative/results.txt`
- **Feature Augmentation Study**: `Results/q3/results.txt`
- **Opposite-Corner Study**: `Results/q4/results.txt`

Each results folder contains a validated summary file aligned with the final report.

## Reproduction Instructions

This repository includes `Project.toml` and `Manifest.toml` for dependency management. An IBM ILOG CPLEX installation is required.

### Entry Points
To reproduce the results, use the following Julia scripts:

```julia
include("src/main.jl")
main()
```

```julia
include("src/main_merge.jl")
main_merge()
```

```julia
include("src/main_iterative_algorithm.jl")
main_iterative()
```

```julia
include("src/test_q3.jl")
main_q3()
```

```julia
include("src/main_q4_opposite_corner.jl")
main_q4_opposite_corner()
```
