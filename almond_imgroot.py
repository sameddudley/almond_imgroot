#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 13:54:28 2023

@author: sdudley
"""


####
#for processing of QTL backlight segmented pictures
#does not use parser command line interface, uses regular inputs
####

import argparse
import cv2
import csv
from func_timeout import func_timeout, FunctionTimedOut
import itertools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
#import multiprocessing
import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from skimage.morphology import thin
from skimage.exposure import match_histograms
#from matplotlib.patches import Polygon
import timeit
from wakepy import keep

class GroundTruth:
    def __init__(self, file):
        self.path = os.path.dirname(file)
        self.name = os.path.split(file)[1].split('.')[0]
        self.genotype = self.name.split('_')[0]
        self.block = self.name.split('_')[1]
        self.date = self.name.split('_')[2]
        self.plant = self.name.split('_')[3]
        self.rep = self.name.split('_')[4]
        self.image = self.load(file)
        self.skeleton = self.skeletonize(self.image)
        self.network, self.pos_dict = self.make_initial_network()
        
    def load(self, image_name):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #self.show_image(image)
        return image
    
    def skeletonize(self, image):
        bool_img = image/255
        skeleton = 255*(thin(bool_img).astype(int))
        #self.show_image(skeleton)
        return skeleton
    
    #def show_image(self, image):
        #fig, ax = plt.subplots()
        #ax.set_aspect('equal')
        #ax.matshow(image, cmap='gray')
        #plt.show()
    
    """
    def show_network(self, network, pos_dict):
        degrees = np.array([[node,val] for (node, val) in network.degree])
        leaves = degrees[np.where(degrees[:,1] == 1)[0]][:,0]
        branches = degrees[np.where(degrees[:,1] > 2)[0]][:,0]
        fig, ax = plt.subplots()
        keys, values = pos_dict.keys(), pos_dict.values()
        inverted_dict = {k: np.array([v[1], -1*v[0]]) for k, v in zip(keys, values)}
        nx.draw_networkx_edges(network, inverted_dict)
        nx.draw_networkx_nodes(network, inverted_dict, nodelist = leaves, node_size = 100, node_color = 'g', ax = ax)
        nx.draw_networkx_nodes(network, inverted_dict, nodelist = branches, node_size = 100, node_color = 'r', ax = ax)
        ax.set_aspect('equal')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        #plt.show()
        """
        

    # This one goes first, and calls the trim tree method to create the network, then returns
    #G, the graph for the network. The write_root_data goes next and puts it all in a dataframe
    def make_initial_network(self, pre_trim = 5, min_branch_length = 10):
        #make initial network from the pixels in the skeleton image
        #connect the points within radius 2 and then create the minimum spanning
        #trees for the network
        xs, ys = np.where(self.skeleton > 0)
        pts = np.column_stack((xs, ys))
        pos_dict = dict(zip(range(len(pts)), pts))

        kdt = KDTree(pts)
        edges = kdt.query_pairs(r = 2, output_type = 'ndarray')
        G = nx.Graph()
        G.add_edges_from(edges, weight = 1)
        sub_trees = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        minimum_spanning_sub_trees = [nx.minimum_spanning_tree(sub) for sub in sub_trees]
        
        if len(minimum_spanning_sub_trees) == 1:
            trimmed = self.trim_tree(minimum_spanning_sub_trees[0], pre_trim, min_branch_length)
            if trimmed is not None:
                #self.show_network(trimmed)
                return trimmed, pos_dict
            else:
                print('No network surived')
                return None, None
        else:
            trimmed_trees = []
            for sub in minimum_spanning_sub_trees:
                #trim distance is parameter that can change, we pre-trim to reduce
                #computation time
                trimmed = self.trim_tree(sub, pre_trim, min_branch_length)
                if trimmed is not None:
                    trimmed_trees.append(trimmed)
            if len(trimmed_trees) > 0:
                union = nx.Graph()
                for sub in trimmed_trees:
                    union = nx.union(union, sub)
                    
                #self.show_network(union, pos_dict)
                return union, pos_dict
            else:
                print('No network surived')
                return None, None
  

    def trim_tree(self, network, pre_trim, min_branch_length):
        #remove some leaves to make the network smaller
        #then remove all branches under min branch length
        G = network.copy()
        for trim in range(pre_trim):
            degrees = np.array([[node,val] for (node, val) in G.degree])
            if len(degrees) > 0:
                leaves = degrees[np.where(degrees[:,1] == 1)[0]][:,0]
                for leaf in leaves:
                    G.remove_node(leaf)
                    
        if len(G.nodes) > 1:
            degrees = np.array([[node,val] for (node, val) in G.degree])
            leaves = degrees[np.where(degrees[:,1] == 1)[0]][:,0]
            branches = degrees[np.where(degrees[:,1] > 2)[0]][:,0]
            if len(branches) == 0:
                leaf1, leaf2 = leaves[0], leaves[-1]
                path_length = nx.shortest_path_length(G, source = leaf1, target = leaf2)
                if path_length >= min_branch_length:
                    return G
                else:
                    None
            branch_lengths = []
            for leaf in leaves:
                min_leaf_path = (np.inf, None)
                for branch in branches:
                    length = nx.shortest_path_length(G, source = branch, target = leaf)
                    if length < min_leaf_path[0]:
                        min_leaf_path = (length, branch)
                branch_lengths.append([min_leaf_path[1], leaf, min_leaf_path[0]])
            branch_lengths = np.array(branch_lengths)
            branches_to_trim = branch_lengths[np.where(branch_lengths[:,2] < min_branch_length)[0], 0:2]
            for branch, leaf in branches_to_trim:
                path = nx.shortest_path(G, source = branch, target = leaf)
                for node in path[1:]:
                    if G.has_node(node):
                        G.remove_node(node)
            return G
        
        else:
            return None
    
    
    def write_root_data(self, bin_size, window):
        G = self.network
        degrees = np.array([[node,val] for (node, val) in G.degree])
        if len(degrees) > 0:
            leaves = degrees[np.where(degrees[:,1] == 1)[0]][:,0]
        #print('there are ', len(leaves), ' root tips')
        pos_dict = self.pos_dict
        nodes = list(G.nodes)
        #we hope ada intersection gives the best radius information
        #and we use that to calculate radii by finding the distance from our
        #medial line points and the closest non-root pixel
        dist = cv2.distanceTransform(self.image, maskSize = cv2.DIST_MASK_PRECISE, distanceType = cv2.DIST_L2)
    
        df = []
        for node in nodes:
            my_pixel = pos_dict[node]
            radius = dist[my_pixel[0], my_pixel[1]]
            neighbors = [neighbor for neighbor in G.neighbors(node)]
            if len(neighbors) == 0:
                continue
            else:
                vector = (my_pixel - pos_dict[neighbors[0]])
                norm = np.linalg.norm(vector)
                vector = vector/norm
                #remember the image x and y are "flipped", so we need to flip them here#to get back to global coordinates
                df.append([node, my_pixel[1], -1 * my_pixel[0], vector[1], vector[0], norm, radius])
        
            
        df = pd.DataFrame(data = df, columns = ['Node', 'x', 'y', 'vector x', 'vector y', 'length', 'radius'])
        df['y'] = df['y'] - np.max(df['y']) #normalize y to zero, all values should be negative, so subtract the max (least negative)
        df['normalized_y'] = df['y']/np.min(df['y'])
        df['normalized_length'] = df['length']/np.sum(df['length'])
        df['csa'] = np.pi*df['radius']*df['radius']
        df['surface_area'] = 2*np.pi*df['radius']*df['length']
        df['normalized_surface_area'] = df['surface_area']/np.sum(df['surface_area'])
        df['volume'] = np.pi*df['radius']*df['radius']*df['length']
        vectors = df[['vector x', 'vector y']].to_numpy()
        dots = np.abs(np.dot(vectors, np.array([0,1])))
        df['dots'] = dots
        df['diameter'] = 2*df['radius']
        df['depth_bins'] = df['y']//bin_size
        df['normalized_depth_bins'] = df['depth_bins']/np.min(df['depth_bins'])
        
        file_name = self.name + '.csv'
        df.to_csv(os.path.join(self.path, file_name))
            
        if len(df) > 0:
            length_depth = df.groupby(['depth_bins'])['normalized_length'].sum()
            length_com = np.sum(length_depth * length_depth.index*bin_size)
            n_length_depth = df.groupby(['normalized_depth_bins'])['normalized_length'].sum()
            n_length_com = np.sum(n_length_depth * n_length_depth.index)
            
            area_depth = df.groupby(['depth_bins'])['normalized_surface_area'].sum()
            area_com = np.sum(area_depth * area_depth.index*bin_size)
            n_area_depth = df.groupby(['normalized_depth_bins'])['normalized_surface_area'].sum()
            n_area_com = np.sum(n_area_depth * n_area_depth.index)
            
            max_x, min_x = np.max(df['x']), np.min(df['x'])
            x_extent = max_x - min_x
            max_y, min_y = np.max(df['y']), np.min(df['y'])
            y_extent = max_y - min_y
            aspect = x_extent/y_extent
            degrees = np.array([[node,val] for (node, val) in G.degree])
            root_tips = len(np.where(degrees[:,1] == 1)[0])
            median_root_radius = np.median(df['radius'])
            average_root_angle = np.degrees(np.arccos(np.mean(df['dots'])))
            total_length = np.sum(df['length'])
            total_area = np.sum(df['surface_area'])
            total_mass = np.sum(df['volume'])
            srl = total_length/total_mass
            sra = total_area/total_mass
            mass_per_tip = total_mass/root_tips
            #put summary file here, maybe split name to get the genotype, ID, and timeseries?
            with open(os.path.join(self.path, 'run' + '_summary.csv'), 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([self.genotype, self.block, self.date, self.plant, self.rep,
                                 total_length, total_area, total_mass,
                                 srl, sra, root_tips, mass_per_tip,
                                 median_root_radius, average_root_angle,
                                 length_com, n_length_com, area_com, n_area_com,
                                 max_x, min_x, max_y, min_y,
                                 x_extent, y_extent, aspect, len(leaves)])
            return df
        else:
            print('no root data!')
            

        
def process_image(file_path):
    img = GroundTruth(file_path)
    img.write_root_data(5, 20)
    

#method for running from command line with a regular input not using parser
with keep.running() as k:
    print("Welcome to img_root_groundtruth")
    type_input = input('Type f for single file, or p for directory of images')

    #start the timer
    starttime = timeit.default_timer()

    #to process a whole directory of binary images
    if type_input == "p":
        directory_path = str(input('Enter directory path:'))
        if os.path.exists(directory_path):
            for file in os.listdir(directory_path):
                if file.endswith(".tif") and (file[0] != '.'): #the != operator keeps from processing hidden files
                    print("processing " + file)
                    process_image(os.path.join(directory_path, file))
        else:
            print("directory does not exist " + directory_path)
                    
                    
    #for a single file  
    elif type_input == 'f':
        file_path = str(input('Enter file path to binary images:'))
        if os.path.exists(file_path):
            process_image(file_path)
        else:
            print("file does not exist " + file_path)

    #if an invalid input
    else:
        print('Invalid entry. Enter f or p')

    print("Full run took :", (timeit.default_timer() - starttime)/60, ' minutes')

"""
#method for running from command line with a regular input not using parser
print("Welcome to img_root_groundtruth")
type_input = input('Type f for single file, or p for directory of images')

#start the timer
starttime = timeit.default_timer()

#to process a whole directory of binary images
if type_input == "p":
    directory_path = str(input('Enter directory path:'))
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            if file.endswith(".tif") and (file[0] != '.'): #the != operator keeps from processing hidden files
                print("processing " + file)
                process_image(os.path.join(directory_path, file))
    else:
        print("directory does not exist " + directory_path)
                
                
#for a single file  
elif type_input == 'f':
    file_path = str(input('Enter file path to binary images:'))
    if os.path.exists(file_path):
        process_image(file_path)
    else:
        print("file does not exist " + file_path)

#if an invalid input
else:
    print('Invalid entry. Enter f or p')

print("Full run took :", (timeit.default_timer() - starttime)/60, ' minutes')

"""
        


#Get this to work with command line interface
#The args.file object is a string, but for some reason does not work in the func
#when I explicity define a string outside of the parser, the function works

#####
#####
#this change in name == main is to stop this block of code from running
#change string back to '__main__' to run this block of code again
####
####


if __name__ == 'CHANGETHISLINE__main__':
    parser = argparse.ArgumentParser()
    
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument("-f", "--file",
                      help="path of image file to classify",
                      type = str)
    
    group.add_argument("-p", "--path",
                      help="path of directory, in which all images will be classified",
                      type = str)
    
    args = parser.parse_args()
    
    starttime = timeit.default_timer()
    with keep.running() as k:
        timeout = 60*90 #in seconds, currently 1.5 hours
        
        if args.file is None:
            if os.path.exists(args.path):
                for file in os.listdir(args.path):
                    if file.endswith(".tif"):
                        print("processing " + file)
                        process_image(os.path.join(args.path, file))
                        
                        #try:
                        #    func_timeout(timeout, process_image, args = (os.path.join(args.path, file)))
                        #except FunctionTimedOut:
                        #    print(file, ' could not complete in ', timeout/60, ' minutes')
            else:
                print('Directory does not exist', args.path)
            
            
        else:
            if os.path.exists(args.file):
                if args.file.endswith('.tif'):
                    process_image(args.file)
                    
                    #try:
                    #    func_timeout(timeout, process_image, args = (args.file)) #this line is wrong I think sending an error
                    #except FunctionTimedOut:
                    #    print(file, ' could not complete in ', timeout/60, ' minutes')
                        
            else:
                print('File does not exist ', args.file)
                
    print("Full run took :", (timeit.default_timer() - starttime)/60, ' minutes')
            