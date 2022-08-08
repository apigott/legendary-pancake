### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 0e208c9f-6ee4-45a2-9402-2352738e716c
begin 
	using Pkg
	Pkg.activate("working")
end

# ╔═╡ 4e21f950-06b5-11ed-313a-55ba91b1aba7
begin
	import JuMP
	import PlutoUI
	using PowerModelsDistribution 
	import InfrastructureModels
	import Ipopt
	import Plots
end

# ╔═╡ 2ca4ff58-b86f-44cc-8cb0-94af6d059d6a
pathof(PowerModelsDistribution)

# ╔═╡ 54eb4c3d-b595-4b94-a774-516cdd5f3b1c
const PMD = PowerModelsDistribution

# ╔═╡ 875349a8-bebe-498f-a6d7-6e0e29286ed9
EVs = collect(1:1)

# ╔═╡ 9cd3f4b7-d911-4d53-aeb2-2fbf779bc13b
begin
	timestamps = collect(0:1:10) # timestamps
	D_k = timestamps[2:end].-timestamps[1:end-1] # duration of each timestep
	K = 1:length(D_k) # set of timesteps
end;

# ╔═╡ f244a0e3-84cf-4c7e-a7b1-f29a551bfca6
data_eng = PMD.parse_file(
	"../PowerModelsDistribution.jl/examples/resources/lvtestcase_notrans.dss",
	transformations=[PMD.remove_all_bounds!]
);

# ╔═╡ 95207977-e96f-46bc-83f9-ca45f3f50214
PMD.reduce_lines!(data_eng);

# ╔═╡ 848d6340-455c-4a58-b068-994b0fbebfbc
data_eng["settings"]["sbase_default"] = 1.0*1E3/data_eng["settings"]["power_scale_factor"];

# ╔═╡ f0860ab8-c0ed-43f9-af2a-e6c9c2600562
PMD.add_bus_absolute_vbounds!(
	data_eng,
	phase_lb_pu=0.9,
	phase_ub_pu=1.1,
	neutral_ub_pu=0.1
);

# ╔═╡ f9c63959-2cae-497f-8e5d-100c1dd71195
begin
	# load to which each EV belongs
	load_e = "load".*string.(EVs)
	# bus to which each EV is connected (same as associated load)
	bus_e = [data_eng["load"][id]["bus"] for id in load_e]
	# phase terminal for each EV (same as associated load)
	phase_e = [data_eng["load"][id]["connections"][1] for id in load_e]
end;

# ╔═╡ a6d10799-1b38-4768-af1f-6a394adb40c0
begin
	data_eng["time_series"] = Dict{String, Any}()
	data_eng["time_series"]["normalized_load_profile"] = Dict{String, Any}(
		"replace"=>false,
		"time"=>float(K),
		"values"=>0.2*cos.((pi/2/maximum(K)).*K)
	)

	for (_, load) in data_eng["load"]
		load["time_series"] = Dict(
			"pd_nom"=>"normalized_load_profile",
			"qd_nom"=>"normalized_load_profile"
		)
	end
	
	data_eng["storage"] = Dict{String, Any}()

	# assume values are in MW/MWh
	for e in EVs
		data_eng["storage"]["EV_stor_$e"] = Dict{String, Any}(
			"status" => PMD.ENABLED,
			"bus" => bus_e[e],
			"connections" => [phase_e[e], 4],
			"configuration" => PMD.WYE,
			"energy" => 0.005,
			"qs_ub" => 0.01, # reactive power ub?
			"qs_lb" => -0.01, # reactive power lb?
			"energy_ub" => 0.01, # e_max
			"charge_ub" => 0.005, # p_c_max
			"discharge_ub" => 0.005, # p_d_max
			"charge_efficiency" => 0.97, # eta_c
			"discharge_efficiency" => 0.95, # eta_d
			"rs" => 0.1, # r_sp?
			"xs" => 0, # x_cp?
			"pex" => 0.1, # not sure?
			"qex" => 0.1, # not sure?
			"cm_ub" => 1, # magnitude of complex current
			"sm_ub" => 1, # magnitude of complex power
		)

	end
end

# ╔═╡ 2f996a49-e40a-4741-910c-ef3210065ceb
data_math_mn = transform_data_model(data_eng, multinetwork=true);

# ╔═╡ 81d42809-16e5-4d49-b602-016cd7b369dd
add_start_vrvi!(data_math_mn);

# ╔═╡ 010f13db-fce4-4f30-8887-5d7c9ff1d32d
pm = PMD.instantiate_mc_model(data_math_mn, PMD.ACPUPowerModel, PMD.build_mn_mc_opf; multinetwork=true);

# ╔═╡ 29c6656c-855b-49a1-a178-c7a5e9d78532
res = solve_mn_mc_opf(data_eng, ACPUPowerModel, Ipopt.Optimizer, setting=Dict("output"=>Dict("duals"=>true)));

# ╔═╡ fca19504-074a-48f3-aba6-a6209eaa7c70
begin
	ses = [res["solution"]["nw"][string(k)]["storage"]["EV_stor_1"]["se"] for k in nw_ids(pm)]
	pss = [res["solution"]["nw"][string(k)]["storage"]["EV_stor_1"]["ps"][1] for k in nw_ids(pm)]
	busps = [[res["solution"]["nw"][string(k)]["load"]["load$l"]["pd_bus"][1] for k in nw_ids(pm)] for l in collect(1:55)]
	loadps = [[res["solution"]["nw"][string(k)]["load"]["load$l"]["pd"][1] for k in nw_ids(pm)] for l in collect(1:55)]
end;

# ╔═╡ e78cf659-fe24-49e7-a9fb-e00d3d3bf90b
begin
	Plots.plot(legend=:none, title="", xlabel="time [h]")
	for e in EVs
		Plots.plot!(timestamps[1:10], pss, markershape=:circle, markersize=3, legend=:topright, labels="P (MW)", left_margin=5Plots.mm, right_margin=15Plots.mm)
	end
	subplot = Plots.twinx()
	Plots.plot!(subplot, timestamps[1:10], ses, markershape=:circle, markersize=3, legend=:bottomright, labels="SE = ?", left_margin=5Plots.mm, right_margin=15Plots.mm)
	Plots.plot!()
end

# ╔═╡ bf50eaef-a23f-4ccc-a162-39c12159523f
begin
	Plots.plot(legend=:none, title="", xlabel="time [h]", ylabel="P_load [kW]")
	for e in EVs
		Plots.plot!(timestamps[1:10], loadps, markershape=:circle, markersize=3)
	end
	Plots.plot!()
end

# ╔═╡ ccdee275-7e25-4558-b3f0-17515c3991ab
begin
	Plots.plot(legend=:none, title="", xlabel="time [h]", ylabel="E [kW]")
	for e in EVs
		Plots.plot!(timestamps[1:10], data_eng["time_series"]["normalized_load_profile"]["values"], markershape=:circle, markersize=3)
	end
	Plots.plot!()
end

# ╔═╡ Cell order:
# ╠═0e208c9f-6ee4-45a2-9402-2352738e716c
# ╠═2ca4ff58-b86f-44cc-8cb0-94af6d059d6a
# ╠═4e21f950-06b5-11ed-313a-55ba91b1aba7
# ╠═54eb4c3d-b595-4b94-a774-516cdd5f3b1c
# ╠═875349a8-bebe-498f-a6d7-6e0e29286ed9
# ╠═9cd3f4b7-d911-4d53-aeb2-2fbf779bc13b
# ╠═f244a0e3-84cf-4c7e-a7b1-f29a551bfca6
# ╠═95207977-e96f-46bc-83f9-ca45f3f50214
# ╠═848d6340-455c-4a58-b068-994b0fbebfbc
# ╠═f0860ab8-c0ed-43f9-af2a-e6c9c2600562
# ╠═f9c63959-2cae-497f-8e5d-100c1dd71195
# ╠═a6d10799-1b38-4768-af1f-6a394adb40c0
# ╠═2f996a49-e40a-4741-910c-ef3210065ceb
# ╠═81d42809-16e5-4d49-b602-016cd7b369dd
# ╠═010f13db-fce4-4f30-8887-5d7c9ff1d32d
# ╠═29c6656c-855b-49a1-a178-c7a5e9d78532
# ╠═fca19504-074a-48f3-aba6-a6209eaa7c70
# ╠═e78cf659-fe24-49e7-a9fb-e00d3d3bf90b
# ╠═bf50eaef-a23f-4ccc-a162-39c12159523f
# ╠═ccdee275-7e25-4558-b3f0-17515c3991ab
