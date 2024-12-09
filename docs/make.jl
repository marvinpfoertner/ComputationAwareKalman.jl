using Revise
Revise.revise()

using Documenter

import ComputationAwareKalman

makedocs(
    sitename = "ComputationAwareKalman.jl",
    # modules = [ComputationAwareKalman],
    pages = ["Home" => "index.md"],
)
