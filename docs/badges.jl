using Requests


function make_badge_simple(subject::String, status::String, color::String, outfile::String)
    subject = replace(subject, " ", "%20")
    url = "https://img.shields.io/badge/$subject-$status-$color.svg"
    res = get(url)
    open(outfile, "w") do fid
        write(fid, res.data)
    end
end

"""
Make badge based on two numbers
"""
function make_badge(subject::String, a::Int, b::Int, outfile::String)
    limits = Float64[100, 80, 60, 40, 20, 0]
    colors = ["brightgreen", "green", "yellowgreen", "yellow", "orange", "red"]
    picked_color = "lightgray"
    perce = a/b*100
    println("perce: $perce")
    for (i, limit) in enumerate(limits)
        if perce >= limit
            picked_color = colors[i]
            break
        end
    end
    println("color: $picked_color")
    message = "$a/$b"
    make_badge_simple(subject, message, picked_color, outfile)
end

"""
Make badge based on one number
"""
function make_badge(subject::String, a::Int, outfile::ASCIIString)
    limits = Float64[0, 20, 40, 60, 80, 100]
    colors = ["brightgreen", "green", "yellowgreen", "yellow", "orange", "red"]
    picked_color = "red"
    for (i, limit) in enumerate(limits)
        if a <= limit
            picked_color = colors[i]
            break
        end
    end
    println("color: $picked_color")
    a = 100 - a
    if a < 0
        a = 0
    end
    message = "$a"
    make_badge_simple(subject, message, picked_color, outfile)
end
