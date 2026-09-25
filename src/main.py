###########################
# Imports
###########################  

import gurobipy as gp
from gurobipy import GRB 

import matplotlib.pyplot as plt

from datetime import date
import math
import networkx as nx
import csv
import time
import json
import sys
import os

import hess
import labeling
import ordering
import fixing
import separation

from gerrychain import Graph
import geopandas as gpd

def cut_edges(G, plan):
    a = { i : j for j in range(len(plan)) for i in plan[j] }
    return sum( 1 for i,j in G.edges if a[i] != a[j] )

def exhibited_deviation(G, plan):
    ideal = sum( G.nodes[i]['TOTPOP'] for i in G.nodes ) / len(plan)
    return max( abs( ideal - sum( G.nodes[i]['TOTPOP'] for i in district ) ) for district in plan )

# this is the pareto frontier for Iowa county-level graph (cut edges vs. deviation)!
warmstarts = [[[2, 3, 12, 18, 24, 25, 26, 28, 38, 39, 44, 46, 47, 54, 56, 60, 67, 71, 72, 82, 83, 86, 87, 90, 94, 95, 96, 97], [5, 6, 7, 10, 11, 15, 29, 34, 45, 51, 61, 69, 75, 77, 80, 85, 88, 98], [1, 9, 13, 14, 16, 19, 20, 21, 22, 23, 27, 30, 31, 32, 33, 35, 36, 37, 40, 41, 42, 48, 49, 50, 52, 53, 55, 57, 58, 59, 62, 63, 64, 65, 68, 70, 74, 76, 78, 79, 84, 89, 91, 92], [0, 4, 8, 17, 43, 66, 73, 81, 93]], [[0, 5, 7, 9, 13, 24, 25, 30, 32, 34, 36, 44, 47, 50, 51, 56, 57, 61, 63, 64, 69, 70, 72, 74, 77, 80, 81, 82, 87, 90, 91], [1, 2, 14, 17, 19, 20, 21, 22, 23, 27, 31, 33, 35, 37, 40, 41, 42, 43, 49, 52, 53, 55, 58, 59, 62, 65, 66, 68, 71, 73, 76, 78, 79, 84, 89, 92, 93], [4, 6, 8, 10, 12, 16, 18, 29, 46, 48, 60, 86, 94, 96], [3, 11, 15, 26, 28, 38, 39, 45, 54, 67, 75, 83, 85, 88, 95, 97, 98]], [[1, 4, 9, 14, 17, 19, 20, 23, 30, 31, 32, 33, 35, 37, 40, 42, 43, 49, 52, 53, 55, 58, 62, 63, 65, 66, 73, 84, 89, 91, 93], [0, 6, 10, 16, 29, 34, 36, 45, 50, 51, 57, 64, 70, 75, 81, 85, 88, 98], [5, 7, 11, 12, 13, 15, 24, 25, 26, 38, 39, 44, 46, 47, 48, 54, 56, 61, 67, 69, 72, 74, 76, 77, 80, 82, 87, 90, 96, 97], [2, 3, 8, 18, 21, 22, 27, 28, 41, 59, 60, 68, 71, 78, 79, 83, 86, 92, 94, 95]], [[6, 12, 21, 22, 27, 29, 34, 38, 39, 46, 48, 51, 54, 60, 67, 71, 75, 85, 86, 88, 96, 97, 98], [5, 7, 10, 11, 13, 15, 16, 24, 25, 26, 36, 44, 45, 47, 50, 56, 57, 61, 69, 72, 74, 77, 80, 81, 82, 87, 90], [0, 3, 4, 8, 17, 18, 28, 43, 49, 66, 83, 95], [1, 2, 9, 14, 19, 20, 23, 30, 31, 32, 33, 35, 37, 40, 41, 42, 52, 53, 55, 58, 59, 62, 63, 64, 65, 68, 70, 73, 76, 78, 79, 84, 89, 91, 92, 93, 94]], [[4, 6, 7, 10, 15, 16, 18, 29, 34, 36, 49, 50, 51, 56, 69, 75, 77, 83, 88], [11, 24, 25, 26, 28, 39, 44, 45, 47, 54, 61, 67, 72, 82, 85, 87, 90, 97, 98], [3, 8, 12, 17, 38, 46, 48, 78, 93, 94, 95], [0, 1, 2, 5, 9, 13, 14, 19, 20, 21, 22, 23, 27, 30, 31, 32, 33, 35, 37, 40, 41, 42, 43, 52, 53, 55, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68, 70, 71, 73, 74, 76, 79, 80, 81, 84, 86, 89, 91, 92, 96]], [[5, 7, 11, 13, 15, 25, 29, 32, 36, 51, 56, 57, 61, 64, 69, 70, 72, 74, 75, 77, 80, 81, 82, 87, 90], [3, 6, 12, 16, 18, 24, 26, 28, 34, 39, 44, 45, 47, 49, 54, 67, 83, 85, 88, 95, 97, 98], [0, 1, 2, 9, 14, 19, 20, 21, 22, 23, 27, 30, 31, 33, 35, 38, 40, 41, 42, 43, 46, 48, 50, 52, 55, 58, 59, 60, 63, 65, 66, 68, 71, 73, 76, 78, 79, 84, 86, 89, 91, 92, 94, 96], [4, 8, 10, 17, 37, 53, 62, 93]], [[1, 2, 3, 11, 12, 15, 21, 22, 24, 25, 26, 27, 38, 39, 41, 44, 46, 47, 53, 54, 60, 61, 67, 68, 71, 72, 76, 79, 82, 86, 87, 90, 96, 97], [0, 5, 9, 13, 14, 17, 19, 20, 23, 30, 31, 32, 33, 35, 37, 40, 42, 43, 49, 52, 55, 57, 58, 59, 62, 63, 64, 65, 66, 70, 73, 74, 81, 84, 89, 91, 92, 93], [4, 8, 16, 18, 28, 48, 78, 83, 94, 95], [6, 7, 10, 29, 34, 36, 45, 50, 51, 56, 69, 75, 77, 80, 85, 88, 98]], [[3, 7, 11, 15, 24, 25, 28, 39, 44, 47, 54, 56, 61, 67, 72, 82, 83, 87, 90, 98], [5, 6, 13, 19, 29, 30, 32, 34, 35, 36, 40, 45, 51, 52, 55, 57, 64, 69, 70, 74, 75, 77, 80, 81, 85, 88, 89, 91], [0, 1, 2, 9, 12, 14, 16, 18, 20, 21, 22, 23, 26, 27, 31, 33, 38, 41, 42, 43, 46, 48, 49, 50, 58, 59, 60, 63, 65, 66, 68, 71, 73, 76, 78, 79, 84, 86, 92, 94, 95, 96, 97], [4, 8, 10, 17, 37, 53, 62, 93]], [[3, 10, 12, 18, 24, 25, 26, 28, 38, 39, 44, 45, 46, 47, 54, 61, 67, 83, 85, 86, 95, 96, 97, 98], [0, 5, 7, 11, 13, 15, 30, 32, 34, 35, 36, 40, 43, 50, 52, 56, 57, 64, 69, 70, 72, 74, 77, 80, 81, 82, 87, 88, 89, 90, 91], [1, 4, 6, 9, 14, 16, 17, 19, 20, 23, 29, 31, 33, 37, 41, 42, 49, 51, 55, 58, 59, 62, 63, 65, 66, 73, 75, 84], [2, 8, 21, 22, 27, 48, 53, 60, 68, 71, 76, 78, 79, 92, 93, 94]], [[5, 7, 11, 12, 13, 15, 24, 25, 26, 29, 36, 39, 44, 47, 51, 56, 57, 61, 69, 72, 74, 77, 80, 82, 86, 87, 90, 96, 97], [3, 18, 28, 34, 38, 45, 46, 54, 67, 75, 83, 85, 88, 94, 95, 98], [4, 8, 10, 17, 37, 53, 62, 93], [0, 1, 2, 6, 9, 14, 16, 19, 20, 21, 22, 23, 27, 30, 31, 32, 33, 35, 40, 41, 42, 43, 48, 49, 50, 52, 55, 58, 59, 60, 63, 64, 65, 66, 68, 70, 71, 73, 76, 78, 79, 81, 84, 89, 91, 92]], [[6, 15, 25, 28, 34, 45, 47, 61, 72, 83, 85, 88, 98], [1, 2, 3, 9, 12, 14, 18, 20, 21, 22, 23, 24, 26, 27, 31, 33, 38, 39, 41, 42, 44, 46, 48, 54, 58, 59, 60, 65, 67, 68, 71, 76, 78, 79, 84, 86, 92, 94, 95, 96, 97], [0, 5, 7, 11, 13, 16, 19, 29, 30, 32, 35, 36, 40, 43, 49, 50, 51, 52, 55, 56, 57, 63, 64, 66, 69, 70, 73, 74, 75, 77, 80, 81, 82, 87, 89, 90, 91], [4, 8, 10, 17, 37, 53, 62, 93]], [[3, 12, 24, 25, 26, 38, 39, 44, 47, 54, 60, 67, 68, 71, 72, 76, 86, 87, 93, 96, 97, 98], [6, 7, 11, 13, 15, 28, 29, 34, 45, 51, 56, 61, 69, 75, 77, 80, 82, 83, 85, 88, 90, 95], [0, 1, 2, 5, 9, 14, 16, 19, 20, 21, 22, 23, 27, 30, 31, 32, 33, 35, 36, 37, 40, 41, 42, 43, 49, 50, 52, 53, 55, 57, 58, 59, 62, 63, 64, 65, 66, 70, 73, 74, 79, 81, 84, 89, 91, 92], [4, 8, 10, 17, 18, 46, 48, 78, 94]], [[5, 6, 7, 11, 13, 29, 30, 32, 34, 35, 36, 40, 45, 51, 56, 57, 63, 64, 69, 70, 74, 75, 77, 80, 81, 85, 88, 89, 90, 91], [3, 12, 15, 24, 25, 26, 28, 39, 44, 47, 54, 61, 67, 72, 82, 83, 86, 87, 96, 97, 98], [0, 1, 4, 9, 14, 17, 19, 20, 21, 22, 23, 27, 31, 33, 37, 41, 42, 43, 49, 50, 52, 53, 55, 58, 59, 62, 65, 66, 73, 79, 84, 92, 93], [2, 8, 10, 16, 18, 38, 46, 48, 60, 68, 71, 76, 78, 94, 95]], [[6, 7, 11, 15, 29, 34, 45, 50, 51, 56, 61, 72, 75, 77, 82, 83, 85, 87, 88, 90], [0, 1, 5, 9, 13, 14, 17, 19, 20, 23, 27, 30, 31, 32, 33, 35, 36, 37, 40, 42, 43, 49, 52, 53, 55, 57, 58, 59, 62, 63, 64, 65, 66, 69, 70, 73, 74, 79, 80, 81, 84, 89, 91, 92], [3, 4, 10, 16, 18, 24, 25, 26, 28, 38, 39, 44, 47, 54, 67, 95, 97, 98], [2, 8, 12, 21, 22, 41, 46, 48, 60, 68, 71, 76, 78, 86, 93, 94, 96]], [[4, 5, 6, 7, 10, 11, 13, 16, 18, 28, 29, 34, 36, 45, 51, 56, 57, 64, 69, 74, 75, 77, 80, 81, 82, 85, 87, 90], [0, 1, 9, 14, 17, 19, 20, 21, 22, 23, 27, 30, 31, 32, 33, 35, 37, 40, 41, 42, 43, 49, 50, 52, 53, 55, 58, 59, 62, 63, 65, 66, 70, 73, 79, 84, 89, 91, 92, 93], [2, 3, 8, 12, 38, 46, 48, 54, 60, 67, 68, 71, 76, 78, 83, 86, 94, 95, 96], [15, 24, 25, 26, 39, 44, 47, 61, 72, 88, 97, 98]], [[3, 12, 24, 25, 26, 28, 38, 39, 44, 46, 47, 48, 54, 61, 67, 68, 76, 78, 83, 86, 94, 95, 96, 97, 98], [6, 7, 11, 13, 15, 29, 34, 45, 50, 51, 56, 69, 72, 75, 77, 80, 82, 85, 87, 88, 90], [0, 4, 5, 9, 10, 14, 16, 17, 18, 19, 23, 30, 31, 32, 33, 35, 36, 37, 40, 42, 43, 49, 52, 55, 57, 58, 62, 63, 64, 65, 66, 70, 73, 74, 81, 84, 89, 91], [1, 2, 8, 20, 21, 22, 27, 41, 53, 59, 60, 71, 79, 92, 93]], [[3, 12, 24, 25, 26, 28, 38, 39, 44, 46, 47, 48, 54, 61, 67, 68, 76, 78, 83, 86, 94, 95, 96, 97, 98], [6, 7, 10, 11, 15, 34, 45, 51, 56, 69, 72, 75, 77, 80, 82, 85, 87, 88, 90], [4, 8, 16, 17, 18, 49, 53, 93], [0, 1, 2, 5, 9, 13, 14, 19, 20, 21, 22, 23, 27, 29, 30, 31, 32, 33, 35, 36, 37, 40, 41, 42, 43, 50, 52, 55, 57, 58, 59, 60, 62, 63, 64, 65, 66, 70, 71, 73, 74, 79, 81, 84, 89, 91, 92]], [[3, 6, 10, 12, 13, 18, 24, 26, 28, 29, 34, 38, 39, 44, 45, 46, 48, 54, 67, 75, 83, 85, 86, 94, 95, 96, 97, 98], [5, 7, 11, 15, 25, 47, 51, 56, 61, 69, 72, 77, 80, 82, 87, 88, 90], [0, 4, 9, 16, 17, 19, 23, 30, 31, 32, 33, 35, 36, 37, 40, 42, 43, 49, 50, 52, 53, 55, 57, 58, 62, 63, 64, 66, 70, 73, 74, 81, 89, 91, 93], [1, 2, 8, 14, 20, 21, 22, 27, 41, 59, 60, 65, 68, 71, 76, 78, 79, 84, 92]], [[3, 12, 24, 25, 26, 28, 38, 39, 44, 46, 47, 48, 54, 61, 67, 68, 76, 78, 83, 86, 94, 95, 96, 97, 98], [5, 6, 7, 11, 13, 15, 29, 34, 45, 51, 56, 69, 72, 75, 77, 80, 82, 85, 87, 88, 90], [1, 2, 9, 14, 19, 20, 21, 22, 23, 27, 30, 31, 32, 33, 35, 36, 37, 40, 41, 42, 52, 53, 55, 57, 58, 59, 60, 62, 63, 64, 65, 70, 71, 73, 74, 79, 81, 84, 89, 91, 92, 93], [0, 4, 8, 10, 16, 17, 18, 43, 49, 50, 66]], [[5, 6, 7, 11, 13, 15, 29, 34, 45, 51, 56, 69, 72, 75, 77, 80, 82, 85, 87, 88, 90], [0, 4, 9, 16, 17, 19, 23, 30, 31, 32, 33, 35, 36, 37, 40, 42, 43, 49, 50, 52, 53, 55, 57, 58, 62, 63, 64, 66, 70, 73, 74, 81, 89, 91, 93], [3, 10, 12, 18, 24, 25, 26, 28, 38, 39, 44, 46, 47, 48, 54, 61, 67, 83, 86, 94, 95, 96, 97, 98], [1, 2, 8, 14, 20, 21, 22, 27, 41, 59, 60, 65, 68, 71, 76, 78, 79, 84, 92]], [[3, 12, 18, 24, 25, 26, 28, 38, 39, 44, 46, 47, 54, 61, 67, 72, 83, 86, 94, 95, 96, 97, 98], [5, 7, 11, 13, 15, 29, 34, 36, 45, 50, 51, 56, 69, 74, 75, 77, 80, 82, 85, 87, 88, 90], [4, 6, 8, 10, 16, 17, 49, 93], [0, 1, 2, 9, 14, 19, 20, 21, 22, 23, 27, 30, 31, 32, 33, 35, 37, 40, 41, 42, 43, 48, 52, 53, 55, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68, 70, 71, 73, 76, 78, 79, 81, 84, 89, 91, 92]]]

################################################
# Summarize computational results to csv file
################################################ 

from csv import DictWriter
def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)
        
        
################################################
# Writes districting solution to json file
################################################ 

def export_to_json(G, districts, filename):
    with open(filename, 'w') as outfile:
        soln = {}
        soln['nodes'] = []
        for j in range(len(districts)):
            for i in districts[j]:
                soln['nodes'].append({
                        'name': G.nodes[i]["NAME20"],
                        'index': i,
                        'district': j
                        })
        json.dump(soln, outfile)
               

################################################
# Draws districts and saves to png file
################################################ 

def export_to_png(G, df, districts, filename):
    
    assignment = [ -1 for u in G.nodes ]
    
    for j in range(len(districts)):
        for i in districts[j]:
            geoID = G.nodes[i]["GEOID20"]
            for u in G.nodes:
                if geoID == df['GEOID20'][u]:
                    assignment[u] = j
    
    if min(assignment[v] for v in G.nodes) < 0:
        print("Error: did not assign all nodes in district map png.")
    else:
        df['assignment'] = assignment
        my_fig = df.plot(column='assignment').get_figure()
        RESIZE_FACTOR = 3
        my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
        plt.axis('off')
        my_fig.savefig(filename)


################################################
# Draws max B set and saves to png file
################################################ 

def export_B_to_png(G, df, B, filename):
    
    B_geoids = [ G.nodes[i]["GEOID20"] for i in B ]
    df['B'] = [1 if df['GEOID20'][u] in B_geoids else 0 for u in G.nodes]
        
    my_fig = df.plot(column='B').get_figure()
    RESIZE_FACTOR = 3
    my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
    plt.axis('off')
    my_fig.savefig(filename)
    

###########################
# Hard-coded inputs
###########################  

state_codes = {
    'WA': '53', 'DE': '10', 'WI': '55', 'WV': '54', 'HI': '15',
    'FL': '12', 'WY': '56', 'NJ': '34', 'NM': '35', 'TX': '48',
    'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
    'PA': '42', 'AK': '02', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '08',
    'CA': '06', 'AL': '01', 'AR': '05', 'VT': '50', 'IL': '17', 'GA': '13',
    'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '04', 'ID': '16', 'CT': '09',
    'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
    'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
    'SC': '45', 'KY': '21', 'OR': '41', 'SD': '46'
}

number_of_congressional_districts = {
    'WA': 10, 'DE': 1, 'WI': 8, 'WV': 3, 'HI': 2,
    'FL': 27, 'WY': 1, 'NJ': 12, 'NM': 3, 'TX': 36,
    'LA': 6, 'NC': 13, 'ND': 1, 'NE': 3, 'TN': 9, 'NY': 27,
    'PA': 18, 'AK': 1, 'NV': 4, 'NH': 2, 'VA': 11, 'CO': 7,
    'CA': 53, 'AL': 7, 'AR': 4, 'VT': 1, 'IL': 18, 'GA': 14,
    'IN': 9, 'IA': 4, 'MA': 9, 'AZ': 9, 'ID': 2, 'CT': 5,
    'ME': 2, 'MD': 8, 'OK': 5, 'OH': 16, 'UT': 4, 'MO': 8,
    'MN': 8, 'MI': 14, 'RI': 2, 'KS': 4, 'MT': 1, 'MS': 4,
    'SC': 7, 'KY': 6, 'OR': 5, 'SD': 1
}

default_config = {
    'state' : 'OK',
    'level' : 'county',
    'base' : 'labeling',
    'fixing' : True,
    'contiguity' : 'scf',
    'symmetry' : 'orbitope',
    'extended' : True,
    'order' : 'B_decreasing',
    'heuristic' : True,
    'lp': True,
    'deviation' : "8000",
    'warmstart' : False,  # Warm start with the optimal IA county-level cut edges solutions?
    'MIP_timelimit' : "3600"
}

available_config = {
    'state' : { key for key in state_codes.keys() },
    'level' : {'county', 'tract'},
    'base' : {'hess', 'labeling'},
    'fixing' : {True, False},
    'contiguity' : {'none', 'lcut', 'scf', 'shir'},
    'symmetry' : {'default', 'aggressive', 'orbitope'},  # orbitope only for labeling
    'extended' : {True, False},
    'order' : {'none', 'decreasing', 'B_decreasing'},
    'heuristic' : {True, False},
    'lp' : {True, False}, # solve and report root LP bound? (in addition to MIP)
    'warmstart' : {True, False}
    #'deviation' : Any
    #'MIP_timelimit' : Any
}


###############################################
# Read configs/inputs and set parameters
############################################### 

# read configs file and load into a Python dictionary

if len(sys.argv)>1:
    # name your own config file in command line, like this: 
    #       python main.py usethisconfig.json
    # to keep logs of the experiments, redirect to file, like this:
    #       python main.py usethisconfig.json 1>>log_file.txt 2>>error_file.txt
    config_filename = sys.argv[1] 
else:
    config_filename = 'config.json' # default
    
print("Reading config from",config_filename)    
config_filename_wo_extension = config_filename.rsplit('.',1)[0]
configs_file = open(config_filename,'r')
batch_configs = json.load(configs_file)
configs_file.close()

# create directory for results
path = os.path.join("..", "results_for_" + config_filename_wo_extension) 
os.mkdir(path) 

# print results to csv file
today = date.today()
today_string = today.strftime("%Y_%b_%d") # Year_Month_Day, like 2019_Sept_16
results_filename = "../results_for_" + config_filename_wo_extension + "/results_" + config_filename_wo_extension + "_" + today_string + ".csv" 

# prepare csv file by writing column headers
with open(results_filename,'w',newline='') as csvfile:   
    my_fieldnames = ['run','state','level','base','fixing','contiguity','symmetry','extended','order','heuristic','lp','deviation','warmstart'] # configs
    my_fieldnames += ['k','L','U','n','m'] # params
    my_fieldnames += ['heur_obj', 'heur_time', 'heur_iter'] # heuristic info
    my_fieldnames += ['B_q', 'B_size', 'B_time', 'B_timelimit'] # max B info
    my_fieldnames += ['DFixings', 'LFixings', 'UFixings_X', 'UFixings_R', 'ZFixings'] # fixing info
    my_fieldnames += ['LP_obj', 'LP_time'] # root LP info
    my_fieldnames += ['MIP_obj','MIP_bound','MIP_time', 'MIP_timelimit', 'MIP_status', 'MIP_nodes', 'callbacks', 'lazy_cuts', 'connected'] # MIP info
    writer = csv.DictWriter(csvfile, fieldnames = my_fieldnames)
    writer.writeheader()
    
############################################################
# Run experiments for each config in batch_config file
############################################################

for key in batch_configs.keys(): 
      
    # get config and check for errors
    config = batch_configs[key]
    print("In run",key,"using config:",config,end='.')
    for ckey in config.keys():
        if ckey != 'deviation' and ckey != 'MIP_timelimit' and config[ckey] not in available_config[ckey]:
            errormessage = "Error: the config option"+ckey+":"+config[ckey]+"is not known."
            sys.exit(errormessage)
    print("")
    
    # fill-in unspecified configs using default values
    for ckey in available_config.keys():
        if ckey not in config.keys():
            print("Using default value",ckey,"=",default_config[ckey],"since no option was selected.")
            config[ckey] = default_config[ckey]
        
    # initialize dictionary to store this run's results
    result = config
    result['run'] = key            
                   
    # read input data
    state = config['state']
    #code = state_codes[state]
    level = config['level']
    G = Graph.from_json("../data/"+state+"_"+level+".json")
    try:
        G.nodes[i]['TOTPOP']
    except:
        for i in G.nodes:
            G.nodes[i]['TOTPOP'] = G.nodes[i]['P0010001'] 
    DG = nx.DiGraph(G) # bidirected version of G
    df = gpd.read_file("../data/"+state+"_"+level+".shp")      

    # set parameters
    k = number_of_congressional_districts[state]        
    population = [G.nodes[i]['TOTPOP'] for i in G.nodes()]    

    deviation = int( config['deviation'] )
    L = math.ceil(sum(population)/k-deviation)
    U = math.floor(sum(population)/k+deviation)
    # deviation = 0.01
    # L = math.ceil((1-deviation/2)*sum(population)/k)
    # U = math.floor((1+deviation/2)*sum(population)/k)

    # deviation = 0
    # L = math.floor( sum(population) / k )
    # U = math.ceil( sum(population) / k )
    
    print("L =",L,", U =",U,", k =",k)
    result['k'] = k
    result['L'] = L
    result['U'] = U
    result['n'] = G.number_of_nodes()
    result['m'] = G.number_of_edges()
    
    # abort early for trivial or overtly infeasible instances
    maxp = max(population[i] for i in G.nodes)
    if k==1 or maxp>U:
        print("k=",k,", max{ p_v | v in V } =",maxp,", U =",U,end='.')
        sys.exit("Aborting early, either due to trivial instance or overtly infeasible instance.")
           
    # read heuristic solution from external file (?)
    heuristic = config['heuristic']
    if heuristic:
        heuristic_file = open('../data/'+level+"/heuristic/heur_"+state+"_"+level+".json",'r')
        heuristic_dict = json.load(heuristic_file)       
        heuristic_districts = [ [node['index'] for node in heuristic_dict['nodes'] if node['district']==j ] for j in range(k) ]
        result['heur_obj'] = heuristic_dict['obj']
        result['heur_time'] = heuristic_dict['time']
        result['heur_iter'] = heuristic_dict['iterations']
    else:
        heuristic_districts = None
        result['heur_obj'] = 'n/a'
        result['heur_time'] = 'n/a'
        result['heur_iter'] = 'n/a'

    # read heuristic solution from hard-coded warmstart?
    if config['warmstart']:
        assert state == 'IA' and level == 'county', "The hard-coded warm starts are only for Iowa county-level cut edges instance."
        heuristic = True # need this to be True for later warmstart injection
        for plan in warmstarts:
            if exhibited_deviation(G, plan) > deviation + 1e-6:
                continue
            if heuristic_districts is None or cut_edges(G, plan) <= cut_edges(G, heuristic_districts):
                heuristic_districts = [ district for district in plan ]
        result['warmstart'] = cut_edges(G, heuristic_districts)
    else:
        result['warmstart'] = 'n/a'
           
    ############################
    # Build base model
    ############################   
    
    m = gp.Model()
    m._DG = DG
    base = config['base']
    
    if base == 'hess':
        # X[i,j]=1 if vertex i is assigned to (district centered at) vertex j
        m._X = m.addVars(DG.nodes, DG.nodes, vtype=GRB.BINARY)
        hess.add_base_constraints(m, population, L, U, k)
    
    if base == 'labeling':        
        # X[i,j]=1 if vertex i is assigned to district j in {0,1,2,...,k-1}
        m._X = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY)
        if config['symmetry']=='orbitope' or config['contiguity'] in {'scf', 'shir', 'lcut'}:
            m._R = m.addVars(DG.nodes, range(k), vtype=GRB.BINARY)
        labeling.add_base_constraints(m, population, L, U, k)

                
    ############################################      
    # Add (extended?) objective 
    ############################################         
    
    extended = config['extended']
    
    if base == 'hess':
        if extended:
            hess.add_extended_objective(m, G)
        else:
            hess.add_objective(m, G)
               
    if base == 'labeling':
        if extended:
            labeling.add_extended_objective(m, G, k)
        else:
            labeling.add_objective(m, G, k)
            
    
    ####################################   
    # Contiguity constraints
    ####################################      
            
    contiguity = config['contiguity']
    m._callback = None
    m._population = population
    m._U = U
    m._k = k
    m._base = base
    m._numLazyCuts = 0
    m._numCallbacks = 0
    
    if base == 'hess':
        if contiguity == 'shir':
            hess.add_shir_constraints(m)
        elif contiguity == 'scf':
            hess.add_scf_constraints(m, G, extended)
        elif contiguity == 'lcut':
            m.Params.lazyConstraints = 1
            m._callback = separation.lcut_separation_generic
                    
    if base == 'labeling':
        if contiguity == 'shir':
            labeling.add_shir_constraints(m, config['symmetry'])
        elif contiguity == 'scf':
            labeling.add_scf_constraints(m, G, extended, config['symmetry'])
        elif contiguity == 'lcut':
            m.Params.lazyConstraints = 1
            m._callback = separation.lcut_separation_generic 
         
    m.update()
    
    
    ############################################
    # Vertex ordering and max B problem 
    ############################################  
        
    order = config['order']
    
    if order == 'B_decreasing':
        (B, result['B_q'], result['B_time'], result['B_timelimit']) = ordering.solve_maxB_problem(DG, population, L, k, heuristic_districts)
        
        # draw set B on map and save
        fn_B = "../" + "results_for_" + config_filename_wo_extension + "/" + result['state'] + "-" + result['level'] + "_" + str(deviation) + "-maxB.png"       
        export_B_to_png(G, df, B, fn_B)
    else:
        (B, result['B_q'], result['B_time'], result['B_timelimit']) = (list(),'n/a','n/a', 'n/a')
        
    result['B_size'] = len(B)
    
    vertex_ordering = ordering.find_ordering(order, B, DG, population)
    position = ordering.construct_position(vertex_ordering)
    
    print("Vertex ordering =", vertex_ordering)  
    print("Position vector =", position)
    print("Set B =", B)

    ####################################   
    # Symmetry handling
    ####################################    
    
    symmetry = config['symmetry']
    
    if symmetry == 'aggressive':
        m.Params.symmetry = 2
    elif symmetry == 'orbitope':
        if base == 'labeling':
            labeling.add_orbitope_extended_formulation(m, G, k, vertex_ordering)
        else:
            sys.exit("Error: orbitope only available for labeling base model.")     
            
            
    ####################################   
    # Variable fixing
    ####################################    
    
    do_fixing = config['fixing']
    
    if do_fixing and base == 'hess':
        result['DFixings'] = fixing.do_Hess_DFixing(m, G, position)
        result['UFixings_R'] = 'n/a'
        
        if contiguity == 'none':
            result['LFixings'] = fixing.do_Hess_LFixing_without_Contiguity(m, G, population, L, vertex_ordering)
            result['UFixings_X'] = fixing.do_Hess_UFixing_without_Contiguity(m, G, population, U)
        else:
            result['LFixings'] = fixing.do_Hess_LFixing(m, G, population, L, vertex_ordering)
            result['UFixings_X'] = fixing.do_Hess_UFixing(m, DG, population, U, vertex_ordering)         
        
        if extended:
            result['ZFixings'] = fixing.do_Hess_ZFixing(m, G)
        else:
            result['ZFixings'] = 0
                
    
    if do_fixing and base == 'labeling':
        result['DFixings'] = fixing.do_Labeling_DFixing(m, G, vertex_ordering, k)
        
        if contiguity == 'none':
            if symmetry == 'orbitope':
                result['LFixings'] = fixing.do_Labeling_LFixing_without_Contiguity(m, G, population, L, vertex_ordering, k)
            else:
                result['LFixings'] = 0
            (result['UFixings_X'], result['UFixings_R']) = fixing.do_labeling_UFixing_without_Contiguity()
        else:
            result['LFixings'] = fixing.do_Labeling_LFixing(m, G, population, L, vertex_ordering, k)
            (result['UFixings_X'], result['UFixings_R']) = fixing.do_Labeling_UFixing(m, DG, population, U, vertex_ordering, k)
        
        if extended:
            result['ZFixings'] = fixing.do_Labeling_ZFixing(m, G, k)
        else:
            result['ZFixings'] = 0
            
    if not do_fixing:
        result['DFixings'] = 0
        result['UFixings_R'] = 0
        result['LFixings'] = 0
        result['UFixings_X'] = 0
        result['ZFixings'] = 0
            
    
    ######################################################################################
    # Solve root LP? Used only for reporting purposes. Not used for MIP solve.
    ######################################################################################  
    
    if config['lp']:
        r = m.relax() # LP relaxation of MIP model m
        #r.Params.LogToConsole = 0 # keep log to a minimum
        r.Params.Method = 3 # use concurrent LP solver
        r.Params.TimeLimit = 3600 # one-hour time limit for solving LP
        print("To get the root LP bound, now solving a (separate) LP model.")
        
        lp_start = time.time()
        r.optimize()
        lp_end = time.time()
        
        if r.status == GRB.OPTIMAL:
            result['LP_obj'] = '{0:.2f}'.format(r.objVal)
        elif r.status == GRB.TIME_LIMIT:
            result['LP_obj'] = 'TL'
        else:
            result['LP_obj'] = '?'
        result['LP_time'] = '{0:.2f}'.format(lp_end - lp_start)
        
    else:
        result['LP_obj'] = 'n/a'
        result['LP_time'] = 'n/a'
        
    
    ####################################   
    # Inject heuristic warm start
    ####################################    
    
    if heuristic and base == 'hess':
        for district in heuristic_districts:    
            p = min([position[v] for v in district])
            j = vertex_ordering[p]
            for i in district:
                m._X[i,j].start = 1
                    
    if heuristic and base == 'labeling':
        center_positions = [ min( position[v] for v in heuristic_districts[j] ) for j in range(k) ] 
        cplabel = { center_positions[j] : j for j in range(k) }
    
        # what node r will root the new district j? The one with earliest position.
        for j in range(k):
            min_cp = min(center_positions)
            r = vertex_ordering[min_cp]
            old_j = cplabel[min_cp]
            
            for i in heuristic_districts[old_j]:
                m._X[i,j].start = 1
                
            center_positions.remove(min_cp)
                
    
    ####################################   
    # Solve MIP
    ####################################  
    
    result['MIP_timelimit'] = float( config['MIP_timelimit'] )
    m.Params.TimeLimit = result['MIP_timelimit']
    m.Params.Method = 3 # use concurrent method for root LP. Useful for degenerate models
    
    start = time.time()
    m.optimize(m._callback)
    end = time.time()
    result['MIP_time'] = '{0:.2f}'.format(end-start)
    
    result['MIP_status'] = int(m.status)
    result['MIP_nodes'] = int(m.NodeCount)
    result['MIP_bound'] = m.objBound
    result['callbacks'] = m._numCallbacks
    result['lazy_cuts'] = m._numLazyCuts
    result['deviation'] = deviation
    
    # report best solution found
    if m.SolCount > 0:
        result['MIP_obj'] = int(m.objVal)

        if base == 'hess':
            labels = [ j for j in DG.nodes if m._X[j,j].x > 0.5 ]
        else: # base == 'labeling'
            labels = [ j for j in range(k) ]
            
        districts = [ [ i for i in DG.nodes if m._X[i,j].x > 0.5 ] for j in labels]
        print("best solution (found) =",districts)
        
        fn = "../" + "results_for_" + config_filename_wo_extension + "/" + result['state'] + "-" + result['level'] + "-" + result['base'] + "-" + result['contiguity'] + "-" + str(config['warmstart'])
        
        # export solution to .json file
        json_fn = fn + "_" + str(deviation) + ".json"
        export_to_json(G, districts, json_fn)
        
        # export solution to .png file (districting map)
        png_fn = fn + "_" + str(deviation) + ".png"
        export_to_png(G, df, districts, png_fn)
        
        # is solution connected?
        connected = True
        for district in districts:
            if not nx.is_connected(G.subgraph(district)):
                connected = False
        result['connected'] = connected
        
    else:
        result['MIP_obj'] = 'no_solution_found'
        result['connected'] = 'n/a'
        
            
    ####################################   
    # Summarize results of this run to csv file
    ####################################  
    
    append_dict_as_row(results_filename,result,my_fieldnames)
    
