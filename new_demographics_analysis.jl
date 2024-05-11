using CSV
using DataFrames
using StatsPlots
using Plots
using Measures

DEMOGRAPHICS_DATA_PATH = "data/NEWdemographics.csv"
EXCLUDE_DATA_PATH = "data/exclusions.csv"

results_dict = final_results[[0.35, 0.35, 0.3]] # determined best setting of prior 

function is_rule_based(id)
    return argmax(results_dict[id][3]) == 1
end

function is_flexible2(id)
    return (argmax(results_dict[id][3]) == 2)&&(argmax(results_dict[id][4]) == 2)
end

function is_flexible3(id)
    return (argmax(results_dict[id][3]) == 2)&&(argmax(results_dict[id][4]) == 3)
end

function is_flexible4(id)
    return (argmax(results_dict[id][3]) == 2)&&(argmax(results_dict[id][4]) == 4)
end

function is_flexible5(id)
    return (argmax(results_dict[id][3]) == 2)&&(argmax(results_dict[id][4]) == 5)
end

function is_agreement(id)
    return argmax(results_dict[id][3]) == 3
end

function load_demographics_dataset(data_path::String, exclusions::Bool=true)
    #load and apply exclusions to demographics table
    demo_table = CSV.read(data_path, DataFrame)
    rename!(demo_table, [:Response_ID] .=> [:responseID])
    #delete!(demo_table, [1])
    #exclusions
    if exclusions
        exclusion_table = CSV.read(EXCLUDE_DATA_PATH, DataFrame)
        delete!(exclusion_table, [1,2,3,4,nrow(exclusion_table)])
        rename!(exclusion_table, [:Column2] .=> [:responseID])
        select!(exclusion_table, [:excluded, :responseID])
        excludeIDs = [id for id in dropmissing(exclusion_table, disallowmissing=true)[:,:responseID]] #number of excluded ids: 26
        for id in excludeIDs
            table = filter!(:responseID => !=(id), demo_table) 
        end
    end
    return demo_table
end

#################################################################################################

function plot_ages() 
    df = load_demographics_dataset(DEMOGRAPHICS_DATA_PATH)
    age_data = (coalesce.(df[:,:age],0))
    x = Plots.histogram(age_data)
    title!("Age Range of Participants")
    xlabel!("Age")
    ylabel!("Number of Participants")
    display(x)
end

function find_age(responseID)
    demo_table = coalesce.(load_demographics_dataset(DEMOGRAPHICS_DATA_PATH),0)
    try
        return filter(:responseID => ==(responseID), demo_table)[:,:age][1]
    catch
        return -1
    end
end

# make age plots for the three catergories

function build_age_plots()
    # for each catergory of person, show distribution of ages. returns 3 subplots.
    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]
    age_plots = []
    for lis in [rule_based_ids,flexible2_ids,flexible3_ids,flexible4_ids,flexible5_ids,agreement_ids]
        age_plot = [find_age(id) for id in lis[1]]
        println(length(age_plot))
        push!(age_plots, age_plot)
    end
    bin_size = 0:5:100 
    p1 = Plots.histogram(age_plots[1], color= RGBA(160/255, 61/255, 61/255), title="Rule-Based", titlefont = font(12,"Computer Modern"),bins = bin_size, normalize=:probability, ylabel="Percentage of Participants Within Type")
    p2 = Plots.histogram(age_plots[2], color= RGBA(39/255, 102/255, 20/255), title="Resource Rational: 10²",titlefont = font(12,"Computer Modern"),bins = bin_size, normalize=:probability)
    p3 = Plots.histogram(age_plots[3], color= RGBA(77/255, 159/255, 52/255),title="Resource Rational: 10³",titlefont = font(12,"Computer Modern"),bins = bin_size, normalize=:probability)
    p4 = Plots.histogram(age_plots[4], color= RGBA(116/255, 202/255, 90/255),title="Resource Rational: 10⁴", titlefont = font(12,"Computer Modern"),bins = bin_size, normalize=:probability)
    p5 = Plots.histogram(age_plots[5], color= RGBA(151/255, 243/255, 122/255), title="Resource Rational: 10⁵", titlefont = font(12,"Computer Modern"),bins = bin_size, normalize=:probability)
    p6 = Plots.histogram(age_plots[6], title="Agreement-Based", titlefont = font(12,"Computer Modern"),bins = bin_size, normalize=:probability)
    Plots.plot(p1, p2, p3, p4, p5, p6, layout=(1, 6), legend=false, xlabel="Ages", size=(1400,350),margin=7mm)
    xlims!(0, 80)
    ylims!(0, 0.45)
end

#build_age_plots()

ages = [find_age(id) for id in keys(results_dict)]

function build_age_participation_ratio_plots() # Generates graph of interest discussed in analysis 
    # bars for each view, and each will have ratio of the 6 types
    rule_based_frac = []
    flexible2_frac = []
    flexible3_frac = []
    flexible4_frac = []
    flexible5_frac = []
    agreement_frac = []

    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]

    total = Dict()
    for i in [20,30,40,50,60,70]
        total[i] = count(x->(i<x<=(i+10)),[find_age(id) for id in keys(results_dict)])
    end
    print("total: ",total)
    for i in [20,30,40,50]
        push!(rule_based_frac, count(x->(i<x<=(i+10)),[find_age(id) for id in rule_based_ids[1]])/total[i])
        push!(flexible2_frac, count(x->(i<x<=(i+10)),[find_age(id) for id in flexible2_ids[1]])/total[i])
        push!(flexible3_frac, count(x->(i<x<=(i+10)),[find_age(id) for id in flexible3_ids[1]])/total[i])
        push!(flexible4_frac, count(x->(i<x<=(i+10)),[find_age(id) for id in flexible4_ids[1]])/total[i])
        push!(flexible5_frac, count(x->(i<x<=(i+10)),[find_age(id) for id in flexible5_ids[1]])/total[i])
        push!(agreement_frac, count(x->(i<x<=(i+10)),[find_age(id) for id in agreement_ids[1]])/total[i])
    end
    tot = total[60]+total[70]
    push!(rule_based_frac, count(x->(60<x<=75),[find_age(id) for id in rule_based_ids[1]])/tot)
    push!(flexible2_frac, count(x->(60<x<=75),[find_age(id) for id in flexible2_ids[1]])/tot)
    push!(flexible3_frac, count(x->(60<x<=75),[find_age(id) for id in flexible3_ids[1]])/tot)
    push!(flexible4_frac, count(x->(60<x<=75),[find_age(id) for id in flexible4_ids[1]])/tot)
    push!(flexible5_frac, count(x->(60<x<=75),[find_age(id) for id in flexible5_ids[1]])/tot)
    push!(agreement_frac, count(x->(60<x<=75),[find_age(id) for id in agreement_ids[1]])/tot)
    for i in 1:2 #just to make space to shift legend over
        push!(rule_based_frac, 0)
        push!(flexible2_frac, 0)
        push!(flexible3_frac, 0)
        push!(flexible4_frac, 0)
        push!(flexible5_frac, 0)
        push!(agreement_frac, 0)
    end 

    barplot = StatsPlots.groupedbar(        
        [rule_based_frac flexible2_frac flexible3_frac flexible4_frac flexible5_frac agreement_frac],
        bar_position=:stack, bar_width=0.7, size=(1000,400), 
        xticks=(1:6, ["20s" "30s" "40s" "50s" "60s" "_" "_"]),
        label=["Rule-Based" "Resource Rational: 10²" "Resource Rational: 10³" "Resource Rational: 10⁴" "Resource Rational: 10⁵" "Agreement-Based"], 
        colour = [RGBA(160/255, 61/255, 61/255) RGBA(39/255, 102/255, 20/255) RGBA(77/255, 159/255, 52/255) RGBA(116/255, 202/255, 90/255) RGBA(151/255, 243/255, 122/255) RGBA(82/255, 182/255, 254/255)],
        legend = :bottomright, ylabel = "Percent of Partipants", title = "Percentage of Types Within Age Ranges", titlefontsize=14, xlabel="Age Range", margin=8mm
    )
    display(barplot) 

    return barplot
end

#build_age_participation_ratio_plots()

#################################################################################################

function find_gender(responseID)
    demo_table = coalesce.(load_demographics_dataset(DEMOGRAPHICS_DATA_PATH),0)
    gender = filter(:responseID => ==(responseID), demo_table)[:,:gender][1]
    if gender in ["female","F","FEMALE","Female","woman", " Female", "Female "]
        return "F"
    elseif gender in ["male","MALE","M","m","man","Male", "Male1"]
        return "M"
    else
        println("g: ", gender, " ID: ", responseID)
        return "?"
    end
end

function plot_genders() 
    # plots regular gender distrib of participants
    gender_plot = [find_gender(id) for id in keys(results_dict)]
    values = [count(x->x=="F",gender_plot), count(x->x=="M",gender_plot)]
    x = Plots.bar(["F","M"], values)
    title!("Gender of Participants")
    xlabel!("Gender")
    ylabel!("Number of Participants")
    ylims!(0, 120)
    annotate!(["F","M"], values.+3, values)
    display(x)
end

#plot_genders()

function build_gender_plots()
    size = (1200, 600)
    # for each catergory of person, show distribution of ages. returns 3 subplots.
    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]
    gender_plots = []
    for lis in [rule_based_ids,flexible2_ids,flexible3_ids,flexible4_ids,flexible5_ids,agreement_ids]
        gender_plot = [find_gender(id) for id in lis[1]]
        push!(gender_plots, gender_plot)
    end
    p1 = Plots.bar(["F","M"], [count(x->x=="F",gender_plots[1])/length(gender_plots[1]), count(x->x=="M",gender_plots[1])/length(gender_plots[1])], color= RGBA(160/255, 61/255, 61/255),title="Rule-Based",titlefont = font(12,"Computer Modern"),ylabel="Percentage of Participants Within Type")
    p2 = Plots.bar(["F","M"], [count(x->x=="F",gender_plots[2])/length(gender_plots[2]), count(x->x=="M",gender_plots[2])/length(gender_plots[2])], color= RGBA(39/255, 102/255, 20/255),title="Resource Rational: 10²",titlefont = font(12,"Computer Modern"),)
    p3 = Plots.bar(["F","M"], [count(x->x=="F",gender_plots[3])/length(gender_plots[3]), count(x->x=="M",gender_plots[3])/length(gender_plots[3])], color= RGBA(77/255, 159/255, 52/255),title="Resource Rational: 10³",titlefont = font(12,"Computer Modern"),)
    p4 = Plots.bar(["F","M"], [count(x->x=="F",gender_plots[4])/length(gender_plots[4]), count(x->x=="M",gender_plots[4])/length(gender_plots[4])], color= RGBA(116/255, 202/255, 90/255),title="Resource Rational: 10⁴",titlefont = font(12,"Computer Modern"),)
    p5 = Plots.bar(["F","M"], [count(x->x=="F",gender_plots[5])/length(gender_plots[5]), count(x->x=="M",gender_plots[5])/length(gender_plots[5])], color= RGBA(151/255, 243/255, 122/255),title="Resource Rational: 10⁵",titlefont = font(12,"Computer Modern"),)
    p6 = Plots.bar(["F","M"], [count(x->x=="F",gender_plots[6])/length(gender_plots[6]), count(x->x=="M",gender_plots[6])/length(gender_plots[6])], title="Agreement-Based",titlefont = font(12,"Computer Modern"),)
    g = Plots.plot(p1, p2, p3, p4, p5, p6, layout=(1, 6), legend=false, xlabel="Gender",size=(1400,350),margin=7mm)
    ylims!(0, 1)
    display(g)
end

#build_gender_plots()

function build_gender_ratio_plots()
    size::Tuple = (600, 300)
    # 3 bars, one for each catergory, 2 colors for gender ratio for each catergory
    F_frac = []
    M_frac = []
    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]
    for lis in [rule_based_ids,flexible2_ids,flexible3_ids,flexible4_ids,flexible5_ids,agreement_ids]
        gender_plot = [find_gender(id) for id in lis[1]]
        gender_plot = filter(x -> !isnothing(x), gender_plot)
        total = length(filter(x -> find_gender(x)!="?",lis[1]))
        push!(F_frac, count(x->x=="F",gender_plot)/total)
        push!(M_frac, count(x->x=="M",gender_plot)/total)
    end
    println("F FRAC: ", F_frac, " M FRAC: ", M_frac)
    barplot = StatsPlots.groupedbar(
        [F_frac M_frac],
        bar_position=:stack, bar_width=0.7, size=size, 
        xticks=(1:6, ["Rule-Based" "Resource Rational: 10²" "Resource Rational: 10³" "Resource Rational: 10⁴" "Resource Rational: 10⁵" "Agreement-Based"]),
        label=["F" "M"], margin = 7mm, xrotation=25, ylabel="Percent of Participants", title="Gender Ratios Across Types"
    )
    display(barplot)
    return barplot
end

#build_gender_ratio_plots()

function build_gender_participation_ratio_plots()
    size = (400, 300)
    # two bars, one for F and one for M, and each will have ratio of the 3 types
    rule_based_frac = []
    flexible2_frac = []
    flexible3_frac = []
    flexible4_frac = []
    flexible5_frac = []
    agreement_frac = []

    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]

    total = Dict("F"=>count(x->x=="F",[find_gender(id) for id in keys(results_dict)]),
                "M" => count(x->x=="M",[find_gender(id) for id in keys(results_dict)]))

    for gender in ["F", "M"]
        push!(rule_based_frac, count(x->x==gender,[find_gender(id) for id in rule_based_ids[1]])/total[gender])
        push!(flexible2_frac, count(x->x==gender,[find_gender(id) for id in flexible2_ids[1]])/total[gender])
        push!(flexible3_frac, count(x->x==gender,[find_gender(id) for id in flexible3_ids[1]])/total[gender])
        push!(flexible4_frac, count(x->x==gender,[find_gender(id) for id in flexible4_ids[1]])/total[gender])
        push!(flexible5_frac, count(x->x==gender,[find_gender(id) for id in flexible5_ids[1]])/total[gender])
        push!(agreement_frac, count(x->x==gender,[find_gender(id) for id in agreement_ids[1]])/total[gender])
    end
    for i in 1:2 #just to make space to shift legend over
        push!(rule_based_frac, 0)
        push!(flexible2_frac, 0)
        push!(flexible3_frac, 0)
        push!(flexible4_frac, 0)
        push!(flexible5_frac, 0)
        push!(agreement_frac, 0)
    end
    barplot = StatsPlots.groupedbar(
        [rule_based_frac flexible2_frac flexible3_frac flexible4_frac flexible5_frac agreement_frac],
        bar_position=:stack, bar_width=0.7, size=size, 
        xticks=(1:3, ["Female" "Male" "_" "_"]),
        label=["Rule-Based" "Resource Rational: 10²" "Resource Rational: 10³" "Resource Rational: 10⁴" "Resource Rational: 10⁵" "Agreement-Based"], 
        colour = [RGBA(160/255, 61/255, 61/255) RGBA(39/255, 102/255, 20/255) RGBA(77/255, 159/255, 52/255) RGBA(116/255, 202/255, 90/255) RGBA(151/255, 243/255, 122/255) RGBA(82/255, 182/255, 254/255)],
        legend = :bottomright, ylabel = "Percent of Partipants", title = "Percentage of Types Within Female and Male Participants", titlefontsize=8
    )
    display(barplot)
    return barplot
end

#build_gender_participation_ratio_plots()

#################################################################################################

INCOMES = Dict(1 => "<10k", 2 => "10k", 3 => "20k", 4 => "30k", 5 => "40k", 6 => "50k",
                7 => "60k", 8 => "70k", 9 => "80k", 10 => "90k", 11 => "100k", 
                12 => ">150k", 14 => "No") # NO CHOICE 13

function find_income(responseID)
    demo_table = coalesce.(load_demographics_dataset(DEMOGRAPHICS_DATA_PATH),0)
    return INCOMES[filter(:responseID => ==(responseID), demo_table)[:,:income][1]]
end

function plot_incomes() 
    # plots regular income range of participants
    df = load_demographics_dataset(DEMOGRAPHICS_DATA_PATH)
    incomes = []
    freq = []
    for i in 1:14
        if i != 13
            push!(incomes, INCOMES[i])
            push!(freq, count(x->x==i,df[:,:income]))
            println("INCOME: ", INCOMES[i], "COUNT: ", count(x->x==i,df[:,:income]))
        end
    end
    x = Plots.bar(incomes, freq, title="Income Range")
    title!("Income Range of Participants")
    xlabel!("Income")
    ylabel!("Number of Participants")
    display(x)
end

#plot_incomes() 

function build_income_plots()
    size = (1600, 300)
    # for each catergory of person, show distribution of income. returns 3 subplots.
    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]
    incomes = [INCOMES[i] for i in [1,2,3,4,5,6,7,8,9,10,11,12,14]]
    income_plots = [] #should be 3 lists
    for lis in [rule_based_ids,flexible2_ids,flexible3_ids,flexible4_ids,flexible5_ids,agreement_ids]
        freq = []
        for i in 1:14
            if i != 13
                push!(freq, count(x->x==INCOMES[i],[find_income(id) for id in lis[1]]))
            end
        end
        push!(income_plots, freq)
    end
    p1 = Plots.bar(incomes, income_plots[1]/sum(income_plots[1]), color= RGBA(160/255, 61/255, 61/255),title="Rule-Based", ylabel="Percentage of Participants within Type", ylims=(0,0.4))
    p2 = Plots.bar(incomes, income_plots[2]/sum(income_plots[2]), color= RGBA(39/255, 102/255, 20/255),title="Resource Rational: 10²", ylims=(0,0.4))
    p3 = Plots.bar(incomes, income_plots[3]/sum(income_plots[3]), color= RGBA(77/255, 159/255, 52/255),title="Resource Rational: 10³", ylims=(0,0.4))
    p4 = Plots.bar(incomes, income_plots[4]/sum(income_plots[4]), color= RGBA(116/255, 202/255, 90/255),title="Resource Rational: 10⁴", ylims=(0,0.8))
    p5 = Plots.bar(incomes, income_plots[5]/sum(income_plots[5]), color= RGBA(151/255, 243/255, 122/255),title="Resource Rational: 10⁵", ylims=(0,0.4))
    p6 = Plots.bar(incomes, income_plots[6]/sum(income_plots[6]), title="Agreement-Based")
    m = Plots.plot(p1, p2, p3, p4,p5,p6,layout=(1, 6), legend=false, xrotation=55, xtickfontsize = 5, xlabel="Income Level",
                    size=(1900,500),margin=10mm)
    display(m)
end

#build_income_plots()

function build_income_participation_ratio_plots()
    # bars for each income, and each will have ratio of the 3 types
    rule_based_frac = []
    flexible2_frac = []
    flexible3_frac = []
    flexible4_frac = []
    flexible5_frac = []
    agreement_frac = []

    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]

    total = Dict()
    for i in 1:14
        if i != 13
            total[i] = count(x->x==INCOMES[i],[find_income(id) for id in keys(results_dict)])
        end
    end
    print("total: ",total)
    println([find_income(id) for id in rule_based_ids[1]])
    for income_level in [1,2,3,4,5,6,7,8,9,10,11,12,14]
        push!(rule_based_frac, count(x->x==INCOMES[income_level],[find_income(id) for id in rule_based_ids[1]])/total[income_level])
        push!(flexible2_frac, count(x->x==INCOMES[income_level],[find_income(id) for id in flexible2_ids[1]])/total[income_level])
        push!(flexible3_frac, count(x->x==INCOMES[income_level],[find_income(id) for id in flexible3_ids[1]])/total[income_level])
        push!(flexible4_frac, count(x->x==INCOMES[income_level],[find_income(id) for id in flexible4_ids[1]])/total[income_level])
        push!(flexible5_frac, count(x->x==INCOMES[income_level],[find_income(id) for id in flexible5_ids[1]])/total[income_level])
        push!(agreement_frac, count(x->x==INCOMES[income_level],[find_income(id) for id in agreement_ids[1]])/total[income_level])
    end
    println( rule_based_frac )
    for i in 1:4 #just to make space to shift legend over
        push!(rule_based_frac, 0)
        push!(flexible2_frac, 0)
        push!(flexible3_frac, 0)
        push!(flexible4_frac, 0)
        push!(flexible5_frac, 0)
        push!(agreement_frac, 0)
    end

    barplot = StatsPlots.groupedbar(        
        [rule_based_frac flexible2_frac flexible3_frac flexible4_frac flexible5_frac agreement_frac],
        bar_position=:stack, bar_width=0.7, size=(1600,400), 
        xticks=(1:13, ["<10k" "10k" "20k" "30k" "40k" "50k" "60k" "70k" "80k" "90k" "100k" ">150k" "No Answer" "_" "_" "_" "_"]),
        label=["Rule-Based" "Resource Rational: 10²" "Resource Rational: 10³" "Resource Rational: 10⁴" "Resource Rational: 10⁵" "Agreement-Based"], 
        colour = [RGBA(160/255, 61/255, 61/255) RGBA(39/255, 102/255, 20/255) RGBA(77/255, 159/255, 52/255) RGBA(116/255, 202/255, 90/255) RGBA(151/255, 243/255, 122/255) RGBA(82/255, 182/255, 254/255)],
        legend = :bottomright, ylabel = "Percent of Partipants", title = "Percentage of Types Within Income Levels", titlefontsize=14, xlabel="Income Level", margin=7mm
    )
    display(barplot) 

    return barplot
end
#build_income_participation_ratio_plots()

#################################################################################################

POLITICAL_VIEWS = Dict(11 => "Very Conservative", 12 => "Lean Conservative", 13 => "Moderate",
                        14 => "Lean Liberal", 15 => "Very Liberal")

function plot_views() 
    # plots regular views range of participants
    df = load_demographics_dataset(DEMOGRAPHICS_DATA_PATH)
    views = []
    freq = []
    for i in 11:15
        push!(views, POLITICAL_VIEWS[i])
        push!(freq, count(x->x==i,df[:,:political_view]))
    end
    x = Plots.bar(views, freq, title="Range of Political Views", xrotation=0.45)
    xlabel!("Politcal View")
    ylabel!("Number of Participants")
    display(x)
end

#plot_views() 

function find_view(responseID)
    demo_table = coalesce.(load_demographics_dataset(DEMOGRAPHICS_DATA_PATH),0)
    return POLITICAL_VIEWS[filter(:responseID => ==(responseID), demo_table)[:,:political_view][1]]
end

function build_views_plots()
    size = (1600, 500)
    # for each catergory of person, show distribution of income. returns 3 subplots.
    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]
    views = [POLITICAL_VIEWS[i] for i in 11:15]
    views_plots = [] #should be 3 lists
    for lis in [rule_based_ids,flexible2_ids,flexible3_ids,flexible4_ids,flexible5_ids,agreement_ids]
        freq = []
        for i in 11:15
                push!(freq, count(x->x==POLITICAL_VIEWS[i],[find_view(id) for id in lis[1]]))
        end
        push!(views_plots, freq)
    end
    p1 = Plots.bar(views, views_plots[1]/sum(views_plots[1]), color= RGBA(160/255, 61/255, 61/255),title="Rule-Based", ylabel="Percentage of Participants within Type")
    p2 = Plots.bar(views, views_plots[2]/sum(views_plots[2]), color= RGBA(39/255, 102/255, 20/255),title="Resource Rational: 10²")
    p3 = Plots.bar(views, views_plots[3]/sum(views_plots[3]), color= RGBA(77/255, 159/255, 52/255),title="Resource Rational: 10³")
    p4 = Plots.bar(views, views_plots[4]/sum(views_plots[4]), color= RGBA(116/255, 202/255, 90/255),title="Resource Rational: 10⁴")
    p5 = Plots.bar(views, views_plots[5]/sum(views_plots[5]), color= RGBA(151/255, 243/255, 122/255),title="Resource Rational: 10⁵")
    p6 = Plots.bar(views, views_plots[6]/sum(views_plots[6]), title="Agreement-Based")
    m = Plots.plot(p1, p2, p3, p4, p5, p6, layout=(1, 6), legend=false, xrotation=45, xtickfontsize = 5, xlabel="Political View",
                        size=(1900,500),margin=10mm)
    ylims!(0,0.8)
    display(m)
end

#build_views_plots()


function build_views_participation_ratio_plots()
    # bars for each view, and each will have ratio of the 6 types
    rule_based_frac = []
    flexible2_frac = []
    flexible3_frac = []
    flexible4_frac = []
    flexible5_frac = []
    agreement_frac = []

    rule_based_ids = [filter(k -> is_rule_based(k), keys(results_dict))]
    flexible2_ids = [filter(k -> is_flexible2(k), keys(results_dict))]
    flexible3_ids = [filter(k -> is_flexible3(k), keys(results_dict))]
    flexible4_ids = [filter(k -> is_flexible4(k), keys(results_dict))]
    flexible5_ids = [filter(k -> is_flexible5(k), keys(results_dict))]
    agreement_ids = [filter(k -> is_agreement(k), keys(results_dict))]

    POLITICAL_VIEWS = Dict(11 => "Very Conservative", 12 => "Lean Conservative", 13 => "Moderate",
                        14 => "Lean Liberal", 15 => "Very Liberal")

    total = Dict()
    for i in 11:15
        total[i] = count(x->x==POLITICAL_VIEWS[i],[find_view(id) for id in keys(results_dict)])
    end
    print("total: ",total)
    for view in [11,12,13,14,15]
        push!(rule_based_frac, count(x->x==POLITICAL_VIEWS[view],[find_view(id) for id in rule_based_ids[1]])/total[view])
        push!(flexible2_frac, count(x->x==POLITICAL_VIEWS[view],[find_view(id) for id in flexible2_ids[1]])/total[view])
        push!(flexible3_frac, count(x->x==POLITICAL_VIEWS[view],[find_view(id) for id in flexible3_ids[1]])/total[view])
        push!(flexible4_frac, count(x->x==POLITICAL_VIEWS[view],[find_view(id) for id in flexible4_ids[1]])/total[view])
        push!(flexible5_frac, count(x->x==POLITICAL_VIEWS[view],[find_view(id) for id in flexible5_ids[1]])/total[view])
        push!(agreement_frac, count(x->x==POLITICAL_VIEWS[view],[find_view(id) for id in agreement_ids[1]])/total[view])
    end

    barplot = StatsPlots.groupedbar(        
        [rule_based_frac flexible2_frac flexible3_frac flexible4_frac flexible5_frac agreement_frac],
        bar_position=:stack, bar_width=0.7, size=(1000,400), 
        xticks=(1:7, ["Very Conservative" "Lean Conservative" "Moderate" "Lean Liberal" "Very Liberal"]),
        label=["Rule-Based" "Resource Rational: 10²" "Resource Rational: 10³" "Resource Rational: 10⁴" "Resource Rational: 10⁵" "Agreement-Based"], 
        colour = [RGBA(160/255, 61/255, 61/255) RGBA(39/255, 102/255, 20/255) RGBA(77/255, 159/255, 52/255) RGBA(116/255, 202/255, 90/255) RGBA(151/255, 243/255, 122/255) RGBA(82/255, 182/255, 254/255)],
        legend = :bottomleft, ylabel = "Percent of Partipants", title = "Percentage of Types Within Kinds of Political View", titlefontsize=14, xlabel="Political View", margin=8mm
    )
    display(barplot) 

    return barplot
end

#build_views_participation_ratio_plots()

#################################################################################################

function is_threshold(id, thres)
    return argmax(results_dict[id][4]) == thres
end

nums=[Inf, 1e2, 1e3, 1e4, 1e5, -Inf]

function threshold_plots()
    size = (1600, 500)
    rule_based_ids = filter(k -> is_threshold(k,1), keys(results_dict))
    t1e2_ids = filter(k -> is_threshold(k,2), keys(results_dict))
    t1e3_ids = filter(k -> is_threshold(k,3), keys(results_dict))
    t1e4_ids = filter(k -> is_threshold(k,4), keys(results_dict))
    t1e5_ids = filter(k -> is_threshold(k,5), keys(results_dict))
    agreement_ids = filter(k -> is_threshold(k,6), keys(results_dict))
    threshold_values = ["rule based","1e2","1e3","1e4","1e5","agreement based"]
    freq = [length(rule_based_ids),length(t1e2_ids),length(t1e3_ids),
    length(t1e4_ids),length(t1e5_ids), length(agreement_ids)]
    println(freq/sum(freq))
    p1 = bar(threshold_values, freq/sum(freq), title="Threshold Value Distribution")
    xlabel!("Threshold Value")
    ylabel!("Percentage of Participants")
    ylims!(0,0.4)
    annotate!(threshold_values, (freq/sum(freq)).+0.02, [round.(i/sum(freq);digits=2) for i in freq])
    display(p1)
end

#=plots =[  
plot_ages(),
build_age_plots(),
build_age_participation_ratio_plots(),

plot_genders(),
build_gender_plots(),
build_gender_ratio_plots(),
build_gender_participation_ratio_plots(), 

plot_incomes() ,
build_income_plots(),
build_income_participation_ratio_plots(), 

plot_views() ,
build_views_plots(),
threshold_plots()]

for plot in plots
    plot
    savefig(plot,"PLOT_$plot.png") # save the most recent fig as filename_string (such as "output.png")
end 






