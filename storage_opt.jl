### A Pluto.jl notebook ###
# v0.19.11

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

# ╔═╡ c928bfec-7451-42ac-b6ab-4c2357a24204
K

# ╔═╡ a6d10799-1b38-4768-af1f-6a394adb40c0
begin
	data_eng["time_series"] = Dict{String, Any}()
	data_eng["time_series"]["sine_load_profile"] = Dict{String, Any}(
		"replace"=>false,
		"time"=>float(K),
		"values"=>0.2*cos.((pi/2/maximum(K)).*K)
	)

	

	for (k, load) in data_eng["load"]
		data_eng["time_series"]["rand_load_profile_$k"] = Dict{String, Any}(
			"replace"=>false,
			"time"=>float(K),
			"values"=>[rand([0.15,0.2]) for _ in K]
		)
		load["time_series"] = Dict(
			"pd_nom"=>"rand_load_profile_$k",
			"qd_nom"=>"rand_load_profile_$k"
		)
	end
	
	data_eng["storage"] = Dict{String, Any}()

	# assume values are in kW/kWh
	for e in EVs
		data_eng["storage"]["EV_stor_$e"] = Dict{String, Any}(
			"status" => PMD.ENABLED,
			"bus" => bus_e[e],
			"connections" => [phase_e[e], 4],
			"configuration" => PMD.WYE,
			"energy" => 6,
			"qs_ub" => 1, # reactive power ub?
			"qs_lb" => -1, # reactive power lb?
			"energy_ub" => 13, # e_max
			"charge_ub" => 5, # p_c_max
			"discharge_ub" => 5, # p_d_max
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

# ╔═╡ 933e5bff-ad29-4230-92e2-98dfd4a3361d
data_eng["load"]

# ╔═╡ 2f996a49-e40a-4741-910c-ef3210065ceb
data_math_mn = transform_data_model(data_eng, multinetwork=true);

# ╔═╡ 81d42809-16e5-4d49-b602-016cd7b369dd
add_start_vrvi!(data_math_mn);

# ╔═╡ 010f13db-fce4-4f30-8887-5d7c9ff1d32d
pm = PMD.instantiate_mc_model(data_math_mn, PMD.ACPUPowerModel, PMD.build_mn_mc_opf; multinetwork=true);

# ╔═╡ 29c6656c-855b-49a1-a178-c7a5e9d78532
res = solve_mn_mc_opf(data_eng, ACPUPowerModel, Ipopt.Optimizer, setting=Dict("output"=>Dict("duals"=>true)));

# ╔═╡ d5eac149-a123-475a-91d3-64836e13a901
sum(res["solution"]["nw"][string(1)]["voltage_source"]["source"]["pg"])

# ╔═╡ 26a98b78-7d06-4f98-ae9a-da51c2ea2a89
begin
	bus_l = [data_eng["load"]["load$i"]["bus"] for i in 1:55]
	phase_l = [data_eng["load"]["load$i"]["connections"][1] for i in 1:55]
end

# ╔═╡ 517bdff0-277b-45eb-a093-e9101ad30f43
[res["solution"]["nw"][string(1)]["bus"][bus_l[i]]["lam_kcl_r"][phase_l[i]] for i in 1:55]

# ╔═╡ fca19504-074a-48f3-aba6-a6209eaa7c70
begin
	ses = [res["solution"]["nw"][string(k)]["storage"]["EV_stor_1"]["se"] for k in 1:10]
	scs = [res["solution"]["nw"][string(k)]["storage"]["EV_stor_1"]["sc"] for k in 1:10]
	sds = [res["solution"]["nw"][string(k)]["storage"]["EV_stor_1"]["sd"] for k in 1:10]
	pss = [res["solution"]["nw"][string(k)]["storage"]["EV_stor_1"]["ps"][1] for k in 1:10]
	busps = [[res["solution"]["nw"][string(k)]["load"]["load$l"]["pd_bus"][1] for k in 1:10] for l in collect(1:55)]
	loadps = [[res["solution"]["nw"][string(k)]["load"]["load$l"]["pd"][1] for k in 1:10] for l in collect(1:55)]
	genps = [[sum(res["solution"]["nw"][string(k)]["voltage_source"]["source"]["pg"]) for k in 1:10] for l in collect(1:55)]
	lmps = [[res["solution"]["nw"][string(k)]["bus"][bus_l[i]]["lam_kcl_r"][phase_l[i]] for k in 1:10] for i in 1:55]
end;

# ╔═╡ e78cf659-fe24-49e7-a9fb-e00d3d3bf90b
begin
	Plots.plot(legend=:none, title="", xlabel="time [h]")
	# Plots.plot!(timestamps[1:10], pss, markershape=:circle, markersize=3, legend=:topright, labels="P (kW)", left_margin=5Plots.mm, right_margin=15Plots.mm)
	Plots.plot!(timestamps[1:10], scs, markershape=:circle, markersize=3, legend=:topright, labels="charge (?)", left_margin=5Plots.mm, right_margin=15Plots.mm, color=:green)
	Plots.plot!(timestamps[1:10], sds, markershape=:circle, markersize=3, legend=:topright, labels="discharge (?)", left_margin=5Plots.mm, right_margin=15Plots.mm, color=:red)
	Plots.yaxis!("P_load [kW]")
	subplot = Plots.twinx()
	Plots.plot!(subplot, timestamps[1:10], ses, markershape=:circle, markersize=3, legend=:bottomright, labels="SOC (kWh)", left_margin=5Plots.mm, right_margin=15Plots.mm, color=:cornflowerblue)
	# Plots.plot!(legend=:none, title="", xlabel="time [h]", ylabel=[","SOC"])
	# Plots.yaxis!(subplot,"SOC [kWh]")
	Plots.plot!()
	
end

# ╔═╡ bf50eaef-a23f-4ccc-a162-39c12159523f
begin
	Plots.plot(legend=:none, title="", xlabel="time [h]", ylabel="P_load [kW]")
	Plots.plot!(timestamps[1:10], loadps, markershape=:circle, markersize=3)
	Plots.plot!(timestamps[1:10], genps, markershape=:circle, markersize=3)
	Plots.plot!()
end

# ╔═╡ 04da7682-ec31-48b8-ad3e-55eefda9be0f
begin
	vm_pu_lk = fill(NaN, length(data_eng["load"]), length(K))
	for k in K, l in 1:10
		if l < 7
			bus_id = data_eng["load"]["load$l"]["bus"]
			bus_ind = data_math_mn["bus_lookup"]["1"][bus_id]
			sol_bus = res["solution"]["nw"]["$k"]["bus"][bus_id]
			data_bus = data_eng["bus"][bus_id]
			vbase = data_math_mn["nw"]["$k"]["bus"]["$bus_id"]["vbase"]
			phase = data_eng["load"]["load$l"]["connections"][1]
			ind = findfirst(data_bus["terminals"].==phase)
			vm_pu_lk[l,k] = abs(sol_bus["vm"][ind])/vbase
		end
	end
	
	Plots.plot(xlabel="time [h]", ylabel="load phase voltage [pu]", legend=:none)
	Plots.plot!([timestamps[K[1]], timestamps[K[end]]], [0.9, 0.9], color=:red, linewidth=3)
	Plots.plot!([timestamps[K[1]], timestamps[K[end]]], [1.1, 1.1], color=:red, linewidth=3)
	for k in K
		Plots.scatter!(fill(timestamps[k], length(data_eng["load"])), vm_pu_lk[:,k], markershape=:circle, markersize=3, label="")
	end
	Plots.
	Plots.plot!()
end

# ╔═╡ ee43c65c-10f5-4259-ab6e-231c9db50187
begin
	Plots.plot(legend=:none, title="", xlabel="time [h]", ylabel="LMP (USD/kW)")
	Plots.scatter!(timestamps[1:10], lmps, markershape=:circle, markersize=3)
	Plots.plot!()
end

# ╔═╡ 2e9574d1-d168-4fa2-8167-3199f273a59f


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
# ╠═c928bfec-7451-42ac-b6ab-4c2357a24204
# ╠═a6d10799-1b38-4768-af1f-6a394adb40c0
# ╠═933e5bff-ad29-4230-92e2-98dfd4a3361d
# ╠═2f996a49-e40a-4741-910c-ef3210065ceb
# ╠═81d42809-16e5-4d49-b602-016cd7b369dd
# ╠═010f13db-fce4-4f30-8887-5d7c9ff1d32d
# ╠═29c6656c-855b-49a1-a178-c7a5e9d78532
# ╠═d5eac149-a123-475a-91d3-64836e13a901
# ╠═517bdff0-277b-45eb-a093-e9101ad30f43
# ╠═26a98b78-7d06-4f98-ae9a-da51c2ea2a89
# ╠═fca19504-074a-48f3-aba6-a6209eaa7c70
# ╠═e78cf659-fe24-49e7-a9fb-e00d3d3bf90b
# ╠═bf50eaef-a23f-4ccc-a162-39c12159523f
# ╠═04da7682-ec31-48b8-ad3e-55eefda9be0f
# ╠═ee43c65c-10f5-4259-ab6e-231c9db50187
# ╠═2e9574d1-d168-4fa2-8167-3199f273a59f
