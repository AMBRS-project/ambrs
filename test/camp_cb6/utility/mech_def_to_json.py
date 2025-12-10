import os

def to_json(lines):
    json = \
'''{
    "camp-data" : [
        {
            "name" : "cb6r5_ae7_aq",
            "type" : "MECHANISM",
            "reactions" : [
                {
                    "type" : "EMISSION",
                    "species" : "PCVOC"
                },                    
'''

    oneliner = ''.join([line.strip() for line in lines])
    lines = oneliner.split(';')
    
    read = False
    last_line = False
    for line in lines:
        if 'REACTIONS[CM]' in line:
            read = True
            continue
        elif 'END MECH' in line:
            last_line = True
            
        if read and not last_line:

            rxn = line
            rxn_type = None
                
            if '&' in rxn:
                if '%2' in rxn:
                    rxn_type = 'CMAQ_OH_HNO3'
                elif '%3' in rxn:
                    rxn_type = 'CMAQ_H2O2'
                elif '%H' not in rxn:
                    rxn_type = 'TROE'
            elif '@' in rxn or ('@' not in rxn and '&' not in rxn and '/' not in rxn and '*' not in rxn and '~' not in rxn) :
                rxn_type = 'ARRHENIUS'
            elif '/<' in rxn:
                rxn_type = 'PHOTOLYSIS'

            if rxn_type == 'CMAQ_OH_HNO3':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('%2 #')
                k0, k2, k3 = rate.split('&')
                A0, C0 = k0.split('@')
                A2, C2 = k2.split('@')
                A3, C3 = k3.split('@')
                parameters = \
f'''
                    "k0_A" : {float(A0):.4e},
                    "k0_B" : {0.0000},
                    "k0_C" : {float(C0):.4e},
                    "k2_A" : {float(A2):.4e},
                    "k2_B" : {0.0000},
                    "k2_C" : {float(C2):.4e},
                    "k3_A" : {float(A3):.4e},
                    "k3_B" : {0.0000},
                    "k3_C" : {float(C3):.4e},
'''
            if rxn_type == 'CMAQ_H2O2':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('%3 #')
                k1, k2 = rate.split('&')
                A1, C1 = k1.split('@')
                A2, C2 = k2.split('@')
                parameters = \
f'''
                    "k1_A" : {float(A1):.4e},
                    "k1_B" : {0.0000},
                    "k1_C" : {float(C1):.4e},
                    "k2_A" : {float(A2):.4e},
                    "k2_B" : {0.0000},
                    "k2_C" : {float(C2):.4e},
'''
            if rxn_type == 'TROE':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('#')
                and_split = rate.split('&')
                if len(and_split) == 2:
                    N = 1.0
                    Fc = 0.6
                    k0, kinf = and_split
                elif len(and_split) == 3:
                    Fc = 0.6
                    k0, kinf, N = and_split
                else:
                    k0, kinf, N, Fc = and_split
                at_split = k0.split('@')
                if len(at_split) > 1:
                    A0B0, C0 = at_split
                else:
                    C0 = 0.0
                    A0B0 = at_split[0]
                carat_split = A0B0.split('^')
                if len(carat_split) > 1:
                    A0, B0 = carat_split
                else:
                    B0 = 0.0
                    A0 = carat_split[0]
                at_split = kinf.split('@')
                if len(at_split) > 1:
                    AinfBinf, Cinf = at_split
                else:
                    Cinf = 0.0
                    AinfBinf = at_split[0]
                carat_split = AinfBinf.split('^')
                if len(carat_split) > 1:
                    Ainf, Binf = carat_split
                else:
                    Binf = 0.0
                    Ainf = carat_split[0]
                parameters = \
f'''
                    "k0_A" : {float(A0):.4e},
                    "k0_B" : {float(B0):.4e},
                    "k0_C" : {float(C0):.4e},
                    "kinf_A" : {float(Ainf):.4e},
                    "kinf_B" : {float(Binf):.4e},
                    "kinf_C" : {float(Cinf):.4e},
                    "Fc" : {float(Fc):.4e},
                    "N" : {float(N):.4e},
'''
            if rxn_type == 'ARRHENIUS':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('#')
                ABC = rate.split('@')
                if len(ABC) > 1:
                    AB, C = ABC
                else:
                    AB = ABC[0]
                    C = 0.
                AB_split = AB.split('^')
                if len(AB_split) > 1:
                    A, B = AB_split
                else:
                    if AB.strip() == '8.3-11':
                        AB = '8.3E-11'
                    A = AB
                    B = 0.
                parameters = \
f'''
                    "A" : {float(A):.4e},
                    "B" : {float(B):.4e},
                    "C" : {float(C):.4e},
'''
            if rxn_type == 'PHOTOLYSIS':
                lhs, rhs = rxn.split('=')
                rhs, rate = rhs.split('#')
                parameters = \
f'''
                    "notes" : "{rate}",
'''
            
            if rxn_type not in ['CMAQ_OH_HNO3','CMAQ_H2O2','ARRHENIUS','TROE','PHOTOLYSIS']:
                continue

            reactants = lhs.split('>')[1].split('+')

            reactants_string = ''''''
            for term in reactants:
                split = term.split('*')
                if len(split) > 1:
                    stoich = split[0]
                    reactant = split[1].strip()
                else:
                    stoich = 1.0
                    reactant = split[0].strip()
                reactants_string += \
f'''
                        "{reactant}" : {{}},
'''

            products = rhs.split('+')
            for i, product in enumerate(products):
                split = product.split('-')
                if len(split) > 1:
                    products[i] = split[0]
                    products.insert(i, split[1])
            
            products_string = ''''''
            for term in products:
                split = term.split('*')
                if len(split) > 1:
                    stoich = split[0]
                    product = split[1].strip()
                else:
                    stoich = 1.0
                    product = split[0].strip()
                products_string += \
f'''
                        "{product}" : {{ "yield" : {stoich} }},
'''

            json += \
f'''                {{
                    "type" : "{rxn_type}",
                    {parameters}
                    "reactants" : {{
                        {reactants_string[:-2] + reactants_string[-1]}
                    }},
                    "products" : {{
                        {products_string[:-2] + products_string[-1]}
                    }}
                }},
'''
        if last_line:
            json = json[:-2] + json[-1]
            json += \
'''            ]
        }
    ]
}
'''
            break
    return(json)


file = open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/mech_cb6r5_ae7_aq.def')

mech = file.readlines()

file.close()

mech_json = to_json(mech)

print(mech_json)

with open('/home/dquevedo/AMBRS/ambrs_mam4_cb6r5_ae7_aq/tests/mechanism.json','w') as json:
    json.write(mech_json)