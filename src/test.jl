# This file is a part of JuliaFEM.
# License is MIT: see https://github.com/JuliaFEM/JuliaFEM.jl/blob/master/LICENSE.md

""" JuliaFEM testing routines. """
module Test

using Base.Test

abstract TestResult

type NormalTestResult <: TestResult
    test_function :: Function
    result
end

type CriticalTestResult <: TestResult
    filename :: ASCIIString
    message
end

function TestResult(test_function::Function, result)
    NormalTestResult(test_function, result)
end

function TestResult(filename::ASCIIString, message)
    CriticalTestResult(filename, message)
end

global test_results = []

    
function get_test_functions(func)
    return Function[]
end

""" Return all functions from module with name starting test """
function get_test_functions(mod::Module)
    test_function_names = filter((k) -> startswith(string(k), "test_"), names(mod, true))
    test_function_expressions = map((k) -> :($mod.$k), test_function_names)
    test_functions = map(eval, test_function_expressions)
    return test_functions
end

""" Run tests from some file. """
function run_test(filename::ASCIIString, test_function=nothing)
    info("running tests from $filename")
    test_module = nothing
    try
        test_module = include(filename)
    catch error
        warn("Unable to include file $filename for testing.")
        err = Base.showerror(Base.STDOUT, error)
        push!(test_results, TestResult(filename, "Unable to include file: $err"))
        return
    end
    if isa(test_function, Void)
        test_functions = get_test_functions(test_module)
    else
        test_functions = [eval( :($test_module.$test_function) )]
    end
    if length(test_functions) == 0
        warn("Unable to get test functions for file $filename. Define test functions inside module, look for test_heat.jl for concrete example how to do that.")
        push!(test_results, TestResult(filename, "Unable to find test functions"))
        return
    end
    for test_function in test_functions
        run_test(test_function)
    end
end

""" Run single test set. """
function run_test(test_function::Function)

    function test_handler(r::Base.Test.Success)
        result = TestResult(test_function, r)
        push!(test_results, result)
    end 

    function test_handler(r::Base.Test.Failure)
        push!(test_results, TestResult(test_function, r))
        warn("test failed: $(r.expr)")
        warn("partially evaluated expression: $(r.resultexpr)")
    end 

    function test_handler(r::Base.Test.Error)
        push!(test_results, TestResult(test_function, r))
        warn("Error when testing: $(r.expr)")
    end 

    Base.Test.with_handler(test_handler) do
        info("$test_function():")
        # TODO: print docstring of test function if defined.
        #@doc($test_function)
        try
            test_function()
        catch error
            warn("testing $test_function() stopped for critical error $error")
            Base.showerror(Base.STDOUT, error)
            print("\n")
            warn("cannot continue to test function $test_function")
            push!(test_results, TestResult("$test_function", error))
        end
        info("$test_function(): done.")
    end

end

function print_test_statistics()
    info("################# TEST RESULTS #####################")
    passed = 0
    failed = 0
    errors = 0
    critical = 0
    for result in test_results
        if isa(result, NormalTestResult)
            if isa(result.result, Base.Test.Success)
                passed += 1
                continue
            elseif isa(result.result, Base.Test.Failure)
                failed += 1
                filename, linenum = Base.functionloc(result.test_function)
                functionname = Base.function_name(result.test_function)
                warn("test failed: $(result.result.expr), partially evaluated expression: $(result.result.expr)")
                warn("in function $functionname, file $filename, line $linenum")
                continue
            elseif isa(result.result, Base.Test.Error)
                errors += 1
                filename, linenum = Base.functionloc(result.test_function)
                functionname = Base.function_name(result.test_function)
                warn("error in function: $(result.result.expr)")
                warn("in function $functionname, file $filename, line $linenum")
                continue
            end
        elseif isa(result, CriticalTestResult)
            critical += 1
            warn("Critical error on $(result.filename): $(result.message)")
        end
    end
    info("$passed test passed, $failed test failed, $errors errors, $critical critical failures")
    return passed, failed, errors, critical
end

export @test, run_test, print_test_statistics

end
