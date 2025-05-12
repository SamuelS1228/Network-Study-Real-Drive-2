
import streamlit as st
import pandas as pd
from optimization import optimize
from visualization import plot_network, summary

# Page config
st.set_page_config(page_title="Warehouse Optimizer – Scenarios", page_icon="🏭", layout="wide")
st.title("Warehouse Optimizer — Scenario Workspace")
st.caption("Create and compare scenarios, then run the solver.")

# Session state storage
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}

def render_sidebar(name, scenario):
    with st.sidebar:
        st.header(f"Inputs — {name}")

        # File upload
        up = st.file_uploader("Store demand CSV", key=f"stores_{name}")
        if up: scenario["upload"] = up
        if "upload" in scenario and st.checkbox("Preview store data", key=f"prev_{name}"):
            st.dataframe(pd.read_csv(scenario["upload"]).head())

        # Cost parameters
        def n(key,label,default,fmt="%.4f",**k):
            scenario.setdefault(key,default)
            scenario[key]=st.number_input(label,value=scenario[key],format=fmt,key=f"{name}_{key}",**k)

        n("rate_out_min","Outbound $/lb‑min",0.02)
        n("fixed_cost","Fixed cost $/warehouse",250000.0,"%.0f",step=50000.0)
        n("sqft_per_lb","Sq ft per lb",0.02)
        n("cost_sqft","Variable $/sq ft / yr",6.0,"%.2f")

        # Drive times
        scenario.setdefault("drive_times",False)
        scenario["drive_times"]=st.checkbox("Use real drive times (ORS)",value=scenario["drive_times"],key=f"drive_{name}")
        if scenario["drive_times"]:
            scenario.setdefault("ors_key","")
            scenario["ors_key"]=st.text_input("OpenRouteService API key",value=scenario["ors_key"],key=f"ors_{name}",type="password")

        # Number of warehouses
        scenario.setdefault("auto_k",True)
        scenario["auto_k"]=st.checkbox("Optimize # warehouses",value=scenario["auto_k"],key=f"auto_{name}")
        if scenario["auto_k"]:
            scenario.setdefault("k_rng",(2,5))
            scenario["k_rng"]=st.slider("k range",1,10,scenario["k_rng"],key=f"k_rng_{name}")
            k_vals=range(int(scenario["k_rng"][0]),int(scenario["k_rng"][1])+1)
        else:
            n("k_fixed","# warehouses",3,"%.0f",step=1,min_value=1,max_value=10)
            k_vals=[int(scenario["k_fixed"])]

        # Fixed centers
        st.subheader("Fixed Warehouses (forced)")
        scenario.setdefault("fixed",[[0.0,0.0,False] for _ in range(10)])
        for i in range(10):
            with st.expander(f"Fixed {i+1}",expanded=False):
                lat=st.number_input("Latitude",value=scenario["fixed"][i][1],key=f"{name}_fc_lat{i}",format="%.6f")
                lon=st.number_input("Longitude",value=scenario["fixed"][i][0],key=f"{name}_fc_lon{i}",format="%.6f")
                use=st.checkbox("Use",value=scenario["fixed"][i][2],key=f"{name}_fc_use{i}")
                scenario["fixed"][i]=[lon,lat,use]
        fixed_centers=[[lon,lat] for lon,lat,use in scenario["fixed"] if use]

        # Candidate centers
        st.subheader("Candidate Warehouse Locations")
        cand_file=st.file_uploader("Candidate CSV (Longitude,Latitude)",key=f"cand_{name}")
        if cand_file:
            cand_df=pd.read_csv(cand_file,header=None,names=["Longitude","Latitude"])
            scenario["candidates"]=cand_df[["Longitude","Latitude"]].values.tolist()
            scenario["use_candidates"]=st.checkbox("Restrict to these candidate sites",value=scenario.get("use_candidates",True),key=f"cand_use_{name}")
            if st.checkbox("Preview candidates",key=f"cand_prev_{name}"):
                st.dataframe(cand_df.head())
        else:
            scenario["candidates"]=[]
            scenario["use_candidates"]=False

        # Run solver
        if st.button("Run solver",key=f"run_{name}"):
            if "upload" not in scenario:
                st.warning("Upload store demand CSV.")
            else:
                df=pd.read_csv(scenario["upload"])
                res=optimize(
                    df,
                    k_vals,
                    scenario["rate_out_min"],
                    scenario["sqft_per_lb"],
                    scenario["cost_sqft"],
                    scenario["fixed_cost"],
                    consider_inbound=False,  # simplified
                    inbound_rate_min=0.0,
                    inbound_pts=[],
                    fixed_centers=fixed_centers,
                    rdc_list=[],
                    transfer_rate_min=0.0,
                    use_drive_times=scenario["drive_times"],
                    ors_api_key=scenario.get("ors_key",""),
                    candidate_centers=scenario["candidates"] if scenario["use_candidates"] else None,
                )
                scenario["result"]=res
                st.success("Solver finished.")

# Tabs
scenario_names=list(st.session_state["scenarios"])
tabs=scenario_names+["➕  New scenario"]
tab_refs=st.tabs(tabs)

# New scenario tab
with tab_refs[-1]:
    new_name=st.text_input("Scenario name")
    if st.button("Create") and new_name:
        if new_name in st.session_state["scenarios"]:
            st.warning("Name exists")
        else:
            st.session_state["scenarios"][new_name]={}
            st.experimental_rerun()

# Existing scenarios
for idx,name in enumerate(scenario_names):
    scn=st.session_state["scenarios"][name]
    with tab_refs[idx]:
        st.header(f"Scenario: {name}")
        render_sidebar(name,scn)
        if "result" in scn:
            r=scn["result"]
            plot_network(r["assigned"],r["centers"])
            summary(
                r["assigned"],r["total_cost"],r["out_cost"],0,0,r["wh_cost"],
                r["centers"],r["demand_per_wh"],scn["sqft_per_lb"]
            )
            csv=r["assigned"].to_csv(index=False).encode()
            st.download_button("Download assignment CSV",csv,file_name=f"{name}_assignment.csv")
