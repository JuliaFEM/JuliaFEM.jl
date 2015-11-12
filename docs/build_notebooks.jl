# Automatically run notebooks and generate rst table for results

include("badges.jl")

""" rst file parser to find title, author and factcheck status.
"""
function parse_rst(filename)
    res = Dict("author" => "unknown", "status" => 0, "title" => "", "abstract" => "")
    title_found = false
    open(filename) do fid
        data = readlines(fid)
        for line in data
            line = strip(line)
            if line == ""
                continue
            end
            if !title_found
                res["title"] = line
                title_found = true
            end
            if startswith(line, "Author(s):")
                auth = split(line[12:end], ' ')
                auth = filter(s -> !('@' in s), auth)
                auth = join(auth, ' ')
                res["author"] = auth
            end
            if startswith(line, "Abstract:")  # not working, todo, fixme, ...
                res["abstract"] = line[11:end]
            end
            if startswith(line, "Failed:")
                res["status"] = 1
            end
            if startswith(line, "Failure")
                res["status"] = 1
            end
            m = matchall(r"[-0-9.]+", line)
        end
    end
    return res
end


function run_notebooks()
    k = 0
    results = Dict[]
    cd(dirname(@__FILE__)) do
        for ipynb in readdir("tutorials")
            tic()
            if !endswith(ipynb, "ipynb")
                continue
            end
            runtime = 0
            status = 1
            println("Running notebook tutorials/$ipynb")
            # port = 34211+k  # we're having some weird port issue with zmq
            # k += 1
            try
                run(`timeout 180 runipy -o tutorials/$ipynb --kernel=julia-0.4`)
                status = 0
            catch error
                warn("running notebook failed")
                Base.showerror(Base.STDOUT, error)
            end
            runtime = toc()
            bn = "tutorials/$(ipynb[1:end-6])"
            #try
            #    run(`ipython nbconvert tutorials/$ipynb --to rst --output=$bn`)
            #catch error
            #    warn("unable to convert notebook to rst format")
            #    Base.showerror(Base.STDOUT, error)
            #end

            try
                run(`ipython nbconvert tutorials/$ipynb --to html --output=$bn`)
            catch error
                warn("unable to convert notebook to html format")
                Base.showerror(Base.STDOUT, error)
            end

            try
                run(`ipython nbconvert tutorials/$ipynb --to latex --output=$bn`)
            catch error
                warn("unable to convert notebook to tex format")
                Base.showerror(Base.STDOUT, error)
            end

            try
                run(`lualatex --output-directory=tutorials $bn.tex`)
            catch error
                warn("unable to convert notebook from tex to pdf")
                Base.showerror(Base.STDOUT, error)
            end

            data = Dict("author" => "unknown", "status" => status, "runtime" => runtime,
                        "filename" => ipynb, "last_run" => time(), "description"=>"")
            res = parse_rst("$bn.rst")
            data["description"] = res["title"]
            data["author"] = res["author"]
            # there is two possibilities, either notbook does not run for syntax error (status = 1)
            # or notebook is fine but factcheck returns failed. This checks the latter one.
            if status == 0
                data["status"] = res["status"]
            end
            push!(results, data)
        end
    end
    return results
end

"""
Make rows from results ready to tabular form
"""
function makerows(results)
    rows = ["id" "author" "description" "status" "ipynb" "pdf" "last run" "runtime (s)"]
    for (i, data) in enumerate(results)
        row = ["na" "unknown" "unknown" "unknown" "unknown" "unknown" "na" "na"]
        println(i)
        row[1] = string(i)
        desc = data["description"]
        if desc == ""
            desc = data["filename"]
        end
        row[2] = data["author"]
        fn = data["filename"][1:end-6]
        row[3] = ":doc:`$desc <$fn>`"

        if data["status"] == 0
            row[4] = ".. image:: /badges/notebook-passing.svg"
        else
            row[4] = ".. image:: /badges/notebook-failing.svg"
        end

        fn = data["filename"]
        row[5] = ":download:`ipynb <$fn>`"
        fn = data["filename"][1:end-6]*".pdf"
        row[6] = ":download:`pdf <$fn>`"

        row[7] = Libc.strftime("%Y-%m-%d %H:%M:%S", data["last_run"])
        row[8] = string(round(data["runtime"], 2))
        rows = vcat(rows, row)
    end
    return rows
end


"""
write rst table. rows is array of arrays.

Expected output

    +----+-----------+------------------------------+------------+---------+--------+------+-----+
    | id | author    | description                  | last run   | runtime  | status | html | pdf |
    +====+===========+==============================+============+=========+========+======+=====+
    |  1 | Jukka Aho | This is placeholder notebook | 2015-07-05 | 123.45  | |pass| | html | pdf |
    +----+-----------+------------------------------+------------+---------+--------+------+-----+


"""
function write_rst_table(rows)
    # determine cell lenghts
    lenghts = zeros(Int, size(rows))
    for i=1:size(rows, 1)
        for j=1:size(rows, 2)
            lenghts[i,j] = length(rows[i,j])+3
        end
    end
    #println(lenghts)
    cl = maximum(lenghts, 1)  # maximum column lenghts
    tl = sum(cl)+1  # total table width
    mp = [1 cumsum(cl, 2)+1] # marker points "+"
    println(mp)
    #println(cl)
    #println(tl)

    function sep(l, m; s="+", d="-")
        p = ""
        for j=1:l
            if j in m
                p *= s
            else
                p *= d
            end
        end
        p *= "\n"
        return p
    end

    s = ""
    # write rows
    s *= sep(tl, mp)
    for i = 1:size(rows, 1)
        s *= "|"
        for j=1:size(rows, 2)
            s *= " " * rows[i, j] * " "
            s *= repeat(" ", cl[j]-lenghts[i,j])
            s *= "|"
        end
        s *= "\n"
        if i == 1
            s *= sep(tl, mp; d="=")
        else
            s *= sep(tl, mp)
        end
    end
    return s
end


function main()
    results = run_notebooks()
    notebooks_total = 0
    notebooks_passing = 0
    for result in results
        notebooks_total += 1
        if result["status"] == 0
            notebooks_passing += 1
        end
    end
    rows = makerows(results)
    table = write_rst_table(rows)
    println(table)
    cd(dirname(@__FILE__)) do
        open("tutorials/notebooks.rst", "w") do fid
            write(fid, table)
        end
        make_badge("notebooks", notebooks_passing, notebooks_total, "badges/notebooks-status.svg")
    end
end


main()
