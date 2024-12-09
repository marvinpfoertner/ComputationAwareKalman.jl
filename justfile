default:
    just --list

format:
    julia --project=@JuliaFormatter --eval 'using JuliaFormatter; format(".")'

docs:
    julia --project=docs docs/make.jl

servedocs:
    julia --project=docs -e 'using ComputationAwareKalman, LiveServer; servedocs(include_dirs=["src/"])'
