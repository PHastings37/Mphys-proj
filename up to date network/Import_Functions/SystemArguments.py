"""
This file contains a function that will be imported to the main network code,
in an attempt to neaten up the code.

Rory Farwell and Patrick Hastings 22/03/2022
"""

import numpy
import time
import sys
import os

def get_system_arguments(user_input_array) :
    """
    This function reads in an array of the user inputs and returns the following useful values:
    -- Number of epochs
    -- Check day (in years)
    -- Save filename of loss plots (may be removed in the future as tensorboard is used)
    -- Save filename of network (network will be saved if it is the best performing so far)

    Rory Farwell and Patrick Hastings 22/03/2022
    """

    if len(user_input_array) != 5 :
        print("Error: incorrect user inputs. Please enter in the following format")
        print(" python <Code_filename.py> <Number of epochs> <Check day> <Plot save name.png> <network save filename>")
        return sys.exit(1)
    
    #NUMBER OF EPOCHS
    num_epochs = int(user_input_array[1])

    #CHECK DAY
    user_choice_of_check_day = float(user_input_array[2])

    #WHETHER TO USE FULL DATASET OR PARTIAL DATASET
    print("You are running on the full dataset")

    #FILENAME
    # Create a folder at path "folder path" if one does not already exist
    plot_filename = user_input_array[3]
    print(plot_filename)
    plot_date = time.strftime("%Y_%m_%d")
    plot_time = time.strftime("%H_%M")
    #plot_folder_path = f"/home/ptrickhastings37_gmail_com/data/rory_and_pat_results/loss_plots/{plot_date}/"
    plot_folder_path = f"/home/ptrickhastings37_gmail_com/data/pat_results/{plot_date}/"
    if not os.path.exists(plot_folder_path):
        os.makedirs(plot_folder_path)

    # User input for the filename of the network save
    network_date = time.strftime("%Y_%m_%d")
    network_filename = user_input_array[4]
    network_folder_path = f"/home/rory_farwell1_gmail_com/data/rory_pat_network_saves/{network_date}/"
    if not os.path.exists(network_folder_path):
        os.makedirs(network_folder_path)
    network_filepath = f"{network_folder_path}{network_filename}"
    print(f"The best performing network from this training run will be saved at the following filepath: {network_folder_path}{network_filename}")

    return num_epochs, user_choice_of_check_day, plot_folder_path, network_filepath, plot_filename