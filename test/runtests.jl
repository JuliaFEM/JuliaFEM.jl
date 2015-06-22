using JuliaFEM
using FactCheck
using Logging
@Logging.configure(level=DEBUG)


facts("Testing if somebody used print, println(), @sprint in src directory") do
  # TODO: make better reqular expression. Currently it will match all print words
  lines_with_print = Dict()
  src_dir = "../src"
  src = readdir(src_dir)
  for file_name in src
    fil = open(joinpath(src_dir,file_name),"r")
    for (line_number,line) in enumerate(readlines(fil))
      if ismatch(r"print",line)
        lines_with_print[file_name * ":$line_number"] = "print found in line: $line_number"
      end
    end
    close(fil)
  end
  @fact lines_with_print => isempty "Instead of println() use Logging.jl package"
end

