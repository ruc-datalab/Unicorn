
#***************************************************************************************
# 2:1:7 split gold
entity_alignment_data = { 
    "dbp_yg": ["train", "data/dbp_yg/"],
    "dbp_wd": ["test", "data/dbp_wd/"]
}

# 2:1:7
string_matching_data = { 
    "smurf1":["train", "data/smurf-addr/"],
    "smurf2":["train", "data/smurf-names/"],
    "smurf3":["train", "data/smurf-res/"],
    "smurf4":["test", "data/smurf-prod/"],
    "smurf5":["train", "data/smurf-cit/"]
}

# 3:1:1 directly use
new_deepmatcher_data = { 
    "m1":["train","data/em-wa/"],
    "m2":["test","data/em-ds/"],
    "m3":["train","data/em-fz/"],
    "m4":["train","data/em-ia/"],
    "m5":["train","data/em-beer/"]
    
}

# 2:1:7 
new_schema_matching_data = { 
    "fab": ["train", "data/fabricated_dataset/"],
    "deepmdatasets": ["test","data/DeepMDatasets/"],
}

# 2:1:7 
ontology_matching_data = {
    "cw":["train", "data/Illinois-onm/"]
}

# 2:1:7 
column_type_data = {
    "efth":["train","data/efthymiou/"],
    "t2d":["train", "data/t2d_col_type_anno/"],
    "limaya":["test", "data/Limaye_col_type_anno/"]
}

# 2:1:7 
entity_linking_data = {
    "t2d":["train","data/t2d/"],
    "limaya":["test", "data/Limaye/"]
}