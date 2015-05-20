# This file is a part of Julia. License is MIT: http://julialang.org/license

@doc """

`tests, net_on = choosetests(choices)` selects a set of tests to be
run. `choices` should be a vector of test names; if empty or set to
`["all"]`, all tests are selected.

This function also supports "test collections": specifically, "linalg"
 refers to collections of tests in the correspondingly-named
directories.

Upon return, `tests` is a vector of fully-expanded test names, and
`net_on` is true if networking is available (required for some tests).
""" ->
function choosetests(choices = [])
    testnames = [
        "testing_testing"
    ]

    if Base.USE_GPL_LIBS && 1 == 0
        testnames = [testnames, "fft", "dsp"; ]
    end

    if isdir(joinpath(JULIA_HOME, Base.DOCDIR, "examples"))  && 1 == 0
        push!(testnames, "examples")
    end

    # parallel tests depend on other workers - do them last
    #push!(testnames, "parallel")

    tests = (ARGS==["all"] || isempty(ARGS)) ? testnames : ARGS

    if "linalg" in tests
        # specifically selected case
        filter!(x -> x != "linalg", tests)
        linalgtests = ["linalg1", "linalg2", "linalg3", "linalg4",
                       "linalg/lapack", "linalg/triangular", "linalg/tridiag",
                       "linalg/bidiag", "linalg/diagonal",
                       "linalg/pinv", "linalg/givens", "linalg/cholesky", "linalg/lu",
                       "linalg/symmetric"]
        if Base.USE_GPL_LIBS
            push!(linalgtests, "linalg/arnoldi")
        end
        prepend!(tests, linalgtests)
    end

    net_required_for = []
    net_on = true
    try
        getipaddr()
    catch
        warn("Networking unavailable: Skipping tests [" * join(net_required_for, ", ") * "]")
        net_on = false
    end

    if ccall(:jl_running_on_valgrind,Cint,()) != 0 && "rounding" in tests
        warn("Running under valgrind: Skipping rounding tests")
        filter!(x -> x != "rounding", tests)
    end

    if !net_on
        filter!(x -> !(x in net_required_for), tests)
    end

    tests, net_on
end
