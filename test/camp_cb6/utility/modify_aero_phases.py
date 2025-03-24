import os
import pandas as pd

def to_json(lines):
    
    soa_defn = pd.read_csv('soa_defn.csv')
    poa_species = list(soa_defn.loc[soa_defn['POA'].str.strip()=='T', 'PM Name'].str.strip(" '"))
    soa_species = list(soa_defn.loc[soa_defn['POA'].str.strip()=='F','PM Name'].str.strip(" '"))

    organics = ['SOA','AORGH2O']
    inorganics = ['AH2O','AH3OP']
    inerts = ['AEC','AOTHR','AFE','AAL','ASI','ATI','ACA','AMG','AK','AMN','ACORS','ASOIL','ASEACAT']

    organics_string = ''''''
    inorganics_string = ''''''
    inerts_string = ''''''
    
    read = False
    last_line = False
    for line in lines:
        if '!SPECIES' in line:
            read = True
            continue
        elif '/' in line:
            last_line = True
            
        if read and not last_line:
            
            split_line = line.split(',')
            
            name = split_line[0].split("'")[1]
            if name in ['NUM','SRF']:
                continue

            if name in inerts:
                continue
            elif name in poa_species: # SOA species lumped into 'SOA'
                organics.append(name)
            elif name not in organics and name not in soa_species and name not in inorganics and name not in inerts:
                inorganics.append(name)
        
        if last_line:
            break

    for s in organics:
        organics_string += \
f'''
                "{s}",
'''
    for s in inorganics:
        inorganics_string += \
f'''
                "{s}",
'''
    for s in inerts:
        inerts_string += \
f'''
                "{s}",
'''
    
    json = \
f'''{{
    "camp-data" : [
        {{
            "name" : "organic",
            "type" : "AERO_PHASE",
            "species" : [
                {organics_string[:-2] + organics_string[-1]}
            ]
        }},
        {{
            "name" : "inorganic",
            "type" : "AERO_PHASE",
            "species" : [
                {inorganics_string[:-2] + inorganics_string[-1]}
            ]
        }},
        {{
            "name" : "inert",
            "type" : "AERO_PHASE",
            "species" : [
                {inerts_string[:-2] + inerts_string[-1]}
            ]
        }}
    ]
}}
'''
    return json

file = open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/AE_cb6r5_ae7_aq.nml')

ae = file.readlines()

file.close()

ae_json = to_json(ae)

print(ae_json)

with open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/simple_aerosol_phases.json','w') as json:
    json.write(ae_json)