import csv 
import numpy as np
import os
import pickle
import re
import json
from datetime import datetime
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
import time as time_module
import ast
from collections import OrderedDict
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)

def visualize_all_clustering_results(data, velocity_type='v', save_dir='clustering_plots', clustering_results=None):
    try:
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Prepare data
        X = np.array(data).reshape(-1, 1)
        
        # Set proper label based on velocity type
        label_map = {
            'v': 'Velocity',
            'log_v': 'Log Velocity',
            'sqrt_v': 'Square Root Velocity'
        }
        plot_label = label_map.get(velocity_type, 'Velocity')
        
        # Define all clustering methods
        methods = ['kmeans', 'spectral', 'gmm', 'agglomerative', 'dbscan']
        
        # Create individual plots for each method using the provided results
        for method in methods:
            plt.figure(figsize=(12, 6))
            
            # Plot histogram of data points - remove density=True to show actual counts
            plt.hist(X, bins=50, alpha=0.5, color='gray', label='Frequency Distribution') 
            #, range=(0, 8))
            
            try:
                if clustering_results and method in clustering_results:
                    thresholds = clustering_results[method]
                    
                    threshold_lines = [
                        ('Idle', thresholds['idle_threshold'], 'r'),
                        ('Walk', thresholds['walk_threshold'], 'g'),
                        ('Run', thresholds['run_threshold'], 'b')
                    ]
                    
                    for label, value, color in threshold_lines:
                        plt.axvline(x=value, color=color, linestyle='--', 
                                  label=f'{label} threshold: {value:.2f}')
                
                    plt.title(f'{plot_label} Clusters using {method.upper()} (n={len(X)})')
                    plt.xlabel(f'{plot_label} (m/s)')
                    plt.ylabel('Number of Data Points')  # Changed from 'Density'
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f'{save_dir}/clusters_{method}_{velocity_type}_{timestamp}.svg'
                    plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
                    print(f"Saved {method} clustering plot to {filename}")
            
            except Exception as e:
                print(f"Error plotting {method} clustering: {str(e)}")
                
            finally:
                plt.close()

        # Create comparison plot with lines using the provided results
        if clustering_results:
            plt.figure(figsize=(12, 8))
            
            methods_list = []
            idle_thresholds = []
            walk_thresholds = []
            run_thresholds = []
            
            for method in methods:
                if method in clustering_results:
                    methods_list.append(method.upper())
                    results = clustering_results[method]
                    idle_thresholds.append(results['idle_threshold'])
                    walk_thresholds.append(results['walk_threshold'])
                    run_thresholds.append(results['run_threshold'])
            
            if methods_list:
                x = range(len(methods_list))
                plt.plot(x, idle_thresholds, 'ro-', label='Idle', linewidth=2, markersize=8)
                plt.plot(x, walk_thresholds, 'go-', label='Walk', linewidth=2, markersize=8)
                plt.plot(x, run_thresholds, 'bo-', label='Run', linewidth=2, markersize=8)
                
                for i in x:
                    plt.annotate(f'{idle_thresholds[i]:.2f}', (i, idle_thresholds[i]), 
                                textcoords="offset points", xytext=(0,10), ha='center')
                    plt.annotate(f'{walk_thresholds[i]:.2f}', (i, walk_thresholds[i]), 
                                textcoords="offset points", xytext=(0,10), ha='center')
                    plt.annotate(f'{run_thresholds[i]:.2f}', (i, run_thresholds[i]), 
                                textcoords="offset points", xytext=(0,10), ha='center')
                
                plt.xlabel(f'Clustering Methods')
                plt.ylabel(f'{plot_label} Threshold Values (m/s)')  # Made y-axis label more specific
                plt.title(f'Comparison of {plot_label} Thresholds Across Different Clustering Methods')
                plt.xticks(x, methods_list, rotation=45)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Set x-axis limit to 8
                #plt.xlim(0, 8)

                plt.tight_layout()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_filename = f'{save_dir}/threshold_comparison_{velocity_type}_{timestamp}.svg'
                plt.savefig(comparison_filename, format='svg', bbox_inches='tight', dpi=300)
                print(f"Saved threshold comparison plot to {comparison_filename}")
                plt.close()
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        print(traceback.format_exc())

def perform_agglomerative_clustering(data, n_clusters=3):
    try:
        X = np.array(data).reshape(-1, 1)
        
        # Add standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Try different linkage methods for more variety
        linkage = np.random.choice(['ward', 'complete', 'average'])
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        clustering.fit(X_scaled)
        
        # Calculate cluster centers and transform back
        unique_labels = np.unique(clustering.labels_)
        centers = [np.mean(X[clustering.labels_ == label]) for label in unique_labels]
        centers.sort()
        
        return {
            'idle_threshold': float(centers[0]),
            'walk_threshold': float(centers[1]),
            'run_threshold': float(centers[2])
        }
    except Exception as e:
        logging.error(f"Error in Agglomerative clustering: {str(e)}")
        logging.error(f"Data sample: {data[:5] if len(data) > 5 else data}")
        raise

def perform_dbscan_clustering(data, eps=None, min_samples=5):
    try:
        X = np.array(data).reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dynamically determine eps if not provided
        if eps is None:
            # Use a percentage of the data range as eps
            eps = np.std(X_scaled) * 0.3
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # Count the number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        logging.info(f"DBSCAN found {n_clusters} clusters.")
        
        if n_clusters < 3:
            logging.warning("DBSCAN couldn't identify 3 distinct clusters. Using quantile-based thresholds.")
            centers = np.quantile(X, [0.33, 0.66, 1.0])
        else:
            # Calculate cluster centers, excluding noise points
            centers = []
            for i in range(max(labels) + 1):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centers.append(np.mean(cluster_points))
            
            centers.sort()
            while len(centers) < 3:
                centers.append(np.max(X))
        
        return {
            'idle_threshold': float(centers[0]),
            'walk_threshold': float(centers[1]),
            'run_threshold': float(centers[2])
        }
    except Exception as e:
        logging.error(f"Error in DBSCAN clustering: {str(e)}")
        logging.error(f"Data sample: {data[:5] if len(data) > 5 else data}")
        raise

def perform_spectral_clustering(data, n_clusters=3, max_samples=10000):
    try:
        X = np.array(data).reshape(-1, 1)
        
        # Limit the number of samples
        if len(X) > max_samples:
            print(f"Limiting spectral clustering to {max_samples} samples due to computational constraints.")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform spectral clustering
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=np.random.randint(0, 1000))
        labels = spectral.fit_predict(X_scaled)
        
        # Calculate cluster centers
        centers = [np.mean(X[labels == i]) for i in range(n_clusters)]
        centers.sort()
        
        return {
            'idle_threshold': float(centers[0]),
            'walk_threshold': float(centers[1]),
            'run_threshold': float(centers[2])
        }
    except Exception as e:
        print(f"Error in spectral clustering: {str(e)}")
        print(traceback.format_exc())
        return None

def perform_gmm_clustering(data, n_clusters=3):
    try:
        X = np.array(data).reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Randomize parameters for more variety
        covariance_type = np.random.choice(['full', 'tied', 'diag', 'spherical'])
        random_state = np.random.randint(0, 1000)
        
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=random_state
        )
        gmm.fit(X_scaled)
        
        # Transform centers back to original scale
        means = scaler.inverse_transform(gmm.means_)
        centers = sorted(means.flatten())
        
        return {
            'idle_threshold': float(centers[0]),
            'walk_threshold': float(centers[1]),
            'run_threshold': float(centers[2])
        }
    except Exception as e:
        print(f"Error in GMM clustering: {str(e)}")
        print(traceback.format_exc())
        return None

def perform_clustering(data, method='kmeans', n_clusters=3):
    try:
        X = np.array(data).reshape(-1, 1)
        # Common preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'kmeans':
            model = KMeans(
                n_clusters=n_clusters, 
                random_state=np.random.randint(0, 1000)
            )
            model.fit(X_scaled)
            centers = scaler.inverse_transform(model.cluster_centers_).flatten()
            
        elif method == 'spectral':
            return perform_spectral_clustering(data, n_clusters=n_clusters)
            
        elif method == 'gmm':
            covariance_type = np.random.choice(['full', 'tied', 'diag', 'spherical'])
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type=covariance_type,
                random_state=np.random.randint(0, 1000)
            )
            model.fit(X_scaled)
            centers = scaler.inverse_transform(model.means_).flatten()
            
        elif method == 'agglomerative':
            linkage = np.random.choice(['ward', 'complete', 'average'])
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage
            )
            labels = model.fit_predict(X_scaled)
            centers = [np.mean(X[labels == i]) for i in range(n_clusters)]
            
        elif method == 'dbscan':
            eps = np.std(X_scaled) * 0.3  # Dynamic eps
            model = DBSCAN(eps=eps, min_samples=5)
            labels = model.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 3:
                centers = np.quantile(X, [0.33, 0.66, 1.0])
            else:
                centers = []
                for i in range(max(labels) + 1):
                    cluster_points = X[labels == i]
                    if len(cluster_points) > 0:
                        centers.append(np.mean(cluster_points))
                while len(centers) < 3:
                    centers.append(np.max(X))
        else:
            raise ValueError(f"Invalid clustering method: {method}")

        # Sort centers and ensure we have exactly 3
        centers = np.sort(centers)
        if len(centers) > 3:
            centers = [centers[0], centers[len(centers)//2], centers[-1]]

        return {
            'idle_threshold': float(centers[0]),
            'walk_threshold': float(centers[1]),
            'run_threshold': float(centers[2])
        }
    
    except Exception as e:
        logging.error(f"Error in clustering with method {method}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def perform_clustering_variations(data, method='kmeans', manual_thresholds=None):
    results = {}
    
    for velocity_type in ['v', 'log_v', 'sqrt_v']:
        logging.info(f"Processing {velocity_type} with {method} method")
        
        if manual_thresholds and velocity_type in manual_thresholds:
            results[f'cluster_centers_{velocity_type}'] = manual_thresholds[velocity_type]
            continue
            
        try:
            velocities = [item[velocity_type] for item in data if velocity_type in item]
            if not velocities:
                raise ValueError(f"No data found for {velocity_type}")
                
            logging.info(f"Found {len(velocities)} data points for {velocity_type}")
            logging.debug(f"Sample data for {velocity_type}: {velocities[:5]}")
            
            # Remove outliers before clustering
            velocities = np.array(velocities)
            q1, q3 = np.percentile(velocities, [25, 75])
            iqr = q3 - q1
            valid_mask = (velocities >= q1 - 1.5 * iqr) & (velocities <= q3 + 1.5 * iqr)
            velocities = velocities[valid_mask]
            
            if method in ['kmeans', 'spectral', 'gmm', 'agglomerative', 'dbscan']:
                results[f'cluster_centers_{velocity_type}'] = perform_clustering(
                    velocities, 
                    method=method
                )
            else:
                logging.warning(f"Unknown method {method}, falling back to K-means")
                results[f'cluster_centers_{velocity_type}'] = perform_clustering(
                    velocities, 
                    method='kmeans'
                )
            
            logging.info(f"Clustering results for {velocity_type}: {results[f'cluster_centers_{velocity_type}']}")
            
        except Exception as e:
            logging.error(f"Error in clustering for {velocity_type}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Fallback to default thresholds
            results[f'cluster_centers_{velocity_type}'] = {
                'idle_threshold': 0.1,
                'walk_threshold': 0.5,
                'run_threshold': 1.0
            }
            logging.info(f"Using default thresholds for {velocity_type} due to clustering error")
    
    return results

# Translation table for special characters
translation_table = str.maketrans({
    'ä': 'a',
    'ö': 'o',
    'ü': 'u',
    'ß': 'ss',
    'Ä': 'A',
    'Ö': 'O',
    'Ü': 'U',
    'Ã': 'a',
    '¡': '',
    '\\': ' ',
    '\'': '',
    #'Ã': 'a',
    '¦': 'e'
})

def translate_special_chars(text):
    return text.translate(translation_table)

def convert_time_to_seconds(time_str):
    try:
        # Split the time string into minutes and seconds
        minutes, seconds = map(int, time_str.split(':'))
        # Calculate the total number of seconds
        return (minutes * 60) + seconds
    except ValueError:
        return 0    
    
def convert_time_to_minutes(time_str):
    try:
        # Split the time string into minutes and seconds
        minutes, seconds = map(float, time_str.split(':'))
        # Calculate the total number of seconds
        return round(minutes + (seconds / 60), 2)
    except ValueError:
        return 0 
    
def vector_length(x0, x1, y0, y1):
    return np.sqrt(((x1-x0)**2) + ((y1-y0)**2))

def velocity_variations(x0, y0, x1, y1):
    """Berechnet verschiedene Variationen der Geschwindigkeit"""
    v = vector_length(x0, x1, y0, y1)
    
    return {
        "v": v,
        "log_v": np.log(v + 1),  # +1 um log(0) zu vermeiden
        "sqrt_v": np.sqrt(v)
    }


def process_csv(csv_path, csv_path_playerInfo, csv_path_eventInfo, file_path_reduced_pkl, file_path_reduced_txt, file_path_index_csv, file_path_index_json, file_path_reduced_json, file_path_reduced_json2, file_path_reduced_json3, file_path_grouped_json, file_path_reduced_playerInfo_txt, file_path_reduced_eventInfo_txt, file_path_grouped_json_event, file_path_grouped_json_event_structured, file_path_grouped_json_by_match_time, base_path, file_path_final_json, file_path_final_json_vector_arr_red, chosen_method, use_manual, manual_thresholds):
    #csv_path = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\TBVLemgoLippe_SCMagdeburg_sr_sport_event_42307971_position_data.csv"
    # Reading the CSV file using csv.reader
    try:
        with open(csv_path, 'r') as csvfile:
            print("Opened CSV file") 
            list_reader = csv.reader(csvfile, delimiter=';')  # "," and ";" prüfen in der CSV Datei und hier anpassen
            # Get the headers (first row)
            headers = next(list_reader)
            # Print the headers
            print("Headers:", headers)

            data_list = list(list_reader)
            data_list_array = np.array(data_list)
            print("List data shape", data_list_array.shape)
        
            matrix = data_list_array
            rows = matrix.shape[0]
            columns = matrix.shape[1]
            print("Rows:", rows, "Columns:", columns)
            #print("Data List", data_list)
            print("Data List Array", data_list_array)

            #Change "LemgoBall1" etc. to -> 1, a regular number to do indexing
            # Extract the column with player names
            player_column = data_list_array[:, 5]
            # Apply regular expression to extract the number at the end
            extracted_numbers = np.array([re.findall(r'\d+$', player)[0] for player in player_column])
            # Replace the original column values with the extracted numbers
            data_list_array[:, 5] = extracted_numbers
            print("New data list array", data_list_array) #"league_id" für "Lemgoball1" ist jetzt 1, "Lemgoball2" ist 2 etc.

            # Extract the specific column (e.g., column 0 which contains names)
            names_column = data_list_array[:, 5]
            # Find unique names in the column
            unique_names = np.unique(names_column)
            # Print the unique names
            print("Unique names in the column:", unique_names) 

            p=0
            i=0
            #Create a new array with the size of unique names, input: league team id and team ids
            data_league_team_id = data_list_array[:, [5, 6,]].astype(int)
            print("Data League Team ID", data_league_team_id)
            arr_league_Team_id = np.empty((0, data_league_team_id.shape[1]), dtype=data_league_team_id.dtype)

            #print("unique name 6", unique_names[0])
            unique_names_int = unique_names.astype(int)
            print("Unique names int", unique_names_int, "Length", len(unique_names_int))

            for p in range(len(unique_names_int)): #duch Länge von unique names iterieren
                for i in range(len(data_league_team_id)): #iterate through league id
                    if data_league_team_id[i, 0] == unique_names_int[p]:
                        arr_league_Team_id = np.vstack([arr_league_Team_id, data_league_team_id[i]])
                        break # Exit inner loop when first match is found
            print("League Team ID", arr_league_Team_id, "Length", len(arr_league_Team_id))
            
            #Sort the array by team id
            arr_sorted = arr_league_Team_id[np.argsort(arr_league_Team_id[:,1])]
            print("Sorted array:")
            print(arr_sorted)
            # Create empty lists to store the results for each team
            new_array_1 = []
            new_array_2 = []
            new_array_3 = []
            new_array_4 = []

            # Initialize counters for each team
            counter_1 = 0
            counter_2 = 0
            counter_3 = 0

            # Iterate over the rows of the sorted array and add an unique index for each team
            for i in range(len(arr_sorted)):
                # Check the value in the second column and append to the corresponding list
                if arr_sorted[i, 1] == 1:
                    new_array_1.append(arr_sorted[i])
                    new_array_4.append(100 + counter_1)
                    counter_1 += 1
                elif arr_sorted[i, 1] == 2:
                    new_array_2.append(arr_sorted[i])
                    new_array_4.append(200 + counter_2)
                    counter_2 += 1
                elif arr_sorted[i, 1] == 3:
                    new_array_3.append(arr_sorted[i])
                    new_array_4.append(300 + counter_3)
                    counter_3 += 1

            # Convert the lists to NumPy arrays
            new_array_1 = np.array(new_array_1)
            new_array_2 = np.array(new_array_2)
            new_array_3 = np.array(new_array_3)
            new_array_4 = np.array(new_array_4)
            # Sort the arrays based on the first column
            new_array_1 = new_array_1[np.argsort(new_array_1[:,0])]
            new_array_2 = new_array_2[np.argsort(new_array_2[:,0])]
            new_array_3 = new_array_3[np.argsort(new_array_3[:,0])]

            # Print the new arrays
            print("New Array Team 1:", new_array_1)
            print("New Array Team 2:", new_array_2)
            print("New Array Team 3:", new_array_3)
            print("New Array Team 4:", new_array_4.reshape(-1, 1))

            print("Size of new_array_1:", new_array_1.shape[0])
            print("Size of new_array_2:", new_array_2.shape[0])
            print("Size of new_array_3:", new_array_3.shape[0])
            print("Size of new_array_4:", new_array_4.shape[0])

           # Concatenate the arrays
            final_array = np.concatenate((new_array_1, new_array_2, new_array_3))
            print("Size of final_index_array:", final_array.shape[0])
            final_array = np.concatenate((final_array, new_array_4.reshape(-1, 1)), axis=1)
            # Print the final array
            print("Final index sorted Array:", final_array)

            # Save the final array as a CSV file
            np.savetxt(file_path_index_csv, final_array, delimiter=",", fmt="%s")
            print(f"Final index sorted array saved to {file_path_index_csv}")

            # Define the keys for each column
            keys = ["league_id", "team", "index"]

            # Convert the NumPy array to a list of dictionaries
            index_list = []
            for row in final_array:
                row_dict = {keys[i]: int(row[i]) for i in range(len(keys))}
                index_list.append(row_dict)

            #named_dict = {"league_teams": index_list}

            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Define the version number
            version_number = "1.0.0"

            # Add a comment with the current date and version number to the dictionary
            named_dict = {
                "__comment__": f"This file contains index data. Data generated on {current_date}. Version: {version_number}.",
                "league_teams": index_list
            }

            # Save the list of dictionaries as a JSON file 
            with open(file_path_index_json, 'w') as f:
                json.dump(named_dict, f, indent=4)
            print(f"Final sorted array saved to {file_path_index_json}")

            times_column = data_list_array[:, 0]
            # Find unique names in the column
            unique_times = np.unique(times_column)
            len_unique_times = len(unique_times)
            print("Unique Times", len_unique_times)

            minutes_total_game = len_unique_times / (20*60)
            print("Total minutes of data in file", minutes_total_game)

            i=0
            counter = 0
            counter_1 = 0
            counter_2 = 0
            sec_counter = 0
            min_counter = 0
            increment_value = 0
            #list = []
            arr_reduced_list = []  # Use a list to collect the rows
            # Initialize a new column for the additional values
            match_time_seconds_column = []  # New column 
            match_time_minutes_column = []  # New column

            #reduced data
            reduced_data = data_list_array[:, [0, 4, 5, 6, 8, 9, 10]] #time, league id, group id, x, y, z
            print("Reduced Data", reduced_data)
            arr_reduced = np.empty((0, reduced_data.shape[1]), dtype=reduced_data.dtype)
        
            try:
                for i in range(len(times_column)-1):
                    # Increase counter by 1 every 50 ms
                    if times_column[i+1] > times_column[i]:
                        counter += 1
                        counter_1 += 1
                        
                    #print("Current time column value: ", times_column[i])
                    # Nach 1 sek, Spieler- und Positionsdaten zur Liste hinzufügen. Sofern alle 300 Reihen die Werte genommen, 
                    # nicht sichergestellt, dass genau alle 1 sek Werte genommen werden 
                    if counter > 19:
                        #print("1 second data at i: ", times_column[i], "i-1: ", times_column[i-1])
                        sec_counter += counter / 20
                        #print("Sekunden: ", int(sec_counter))
                        minute = round((sec_counter/60), 2)
                        #print("Minute: ", minute)
                        j = i
                        m = 0
                        increment_value += 1
                        while times_column[j] == times_column[j-1]:
                            #arr_reduced = np.vstack([arr_reduced, reduced_data[j]]) #Stack arrays in sequence vertically (row wise).
                            arr_reduced_list.append(reduced_data[j])  # Use append to add rows to the list
                            #print("Reihe: ", j, "times columns value append: ", reduced_data[j])
                            match_time_seconds_column.append(increment_value)  # Set the new column value to 1
                            match_time_minutes_column.append(round(increment_value/60, 2))
                            j -= 1
                            m += 1
                        #print("Anzahl Reihen", m)
                        counter = 0
                        counter_2 += 1
                    if counter_1 > 299:
                        #print("sec", sec_counter)
                        min_counter = sec_counter / 60
                        print("Minuten: ", int(min_counter))
                        counter_1 = 0

                # Convert the list to a NumPy array at the end
                arr_reduced = np.array(arr_reduced_list)
                match_time_seconds = np.array(match_time_seconds_column)
                match_time_minutes = np.array(match_time_minutes_column)

                # new column added to the array
                arr_reduced = np.column_stack((arr_reduced, match_time_seconds))
                arr_reduced = np.column_stack((arr_reduced, match_time_minutes))

                # Save the dictionary to a file using pickle 
                with open(file_path_reduced_pkl, 'wb') as f:
                    pickle.dump(arr_reduced, f)
                    print("Array saved to file arr_reduced.pkl")

                # Save the array to a text file
                np.savetxt(file_path_reduced_txt, arr_reduced, delimiter=',', fmt='%s'), #, fmt='%s' for string
                print(f"Array saved to {file_path_reduced_txt}")

                # Define the keys for each column
                keys = ["time_ms", "player_name", "league_id", "team", "x", "y", "z", "match_time_seconds_column", "match_time_minutes_column"]

                # Identify which columns contain float values
                float_columns = ["x", "y", "z", "match_time_minutes_column"]

                # Identify the column that contains string values
                string_columns = ["player_name"]

                # Dictionaries to count occurrences
                player_count = {}
                league_id_count = {}

                # Initialize a dictionary to keep track of when each player was last seen
                last_seen = {}

                # Initialize a set to keep track of players we've already added entries for
                added_missing_entries = set()

                # Initialize a dictionary to store player information
                player_info = {}

                # Convert the NumPy array to a list of dictionaries and process the data
                final_reduced_list = []
                for idx, row in enumerate(arr_reduced):
                    row_dict = {}
                    for i, key in enumerate(keys):
                        if key in float_columns:
                            row_dict[key] = float(row[i])
                        elif key in string_columns:
                            translated_value = translate_special_chars(str(row[i]))
                            row_dict[key] = translated_value
                            row[i] = translated_value  # Use translated value for counting
                        else:
                            row_dict[key] = int(row[i])
                
                    # Update the player and league ID counts
                    player_name = row_dict["player_name"]
                    league_id = row_dict["league_id"]
                    match_time_seconds = row_dict["match_time_seconds_column"]

                    # Set "z" value to 0 if league_id is greater than 5
                    if league_id > 5:
                        row_dict["z"] = 0
                    
                    if player_name not in player_count:
                        player_count[player_name] = 0
                    player_count[player_name] += 1

                    if league_id not in league_id_count:
                        league_id_count[league_id] = 0
                    league_id_count[league_id] += 1

                    # Update last_seen for this player and store player info
                    last_seen[league_id] = match_time_seconds
                    player_info[league_id] = {
                        "team": row_dict["team"],
                        "player_name": player_name
                    }

                    final_reduced_list.append(row_dict)
                    z = 0
                    # Check for players missing for 4 consecutive time steps
                    if match_time_seconds >= 5:
                        for player_league_id, last_seen_time in list(last_seen.items()):
                            if match_time_seconds - last_seen_time == 4 and player_league_id not in added_missing_entries:
                                # Player has been missing for 4 time steps and we haven't added an entry yet
                                z += 1
                                missing_player = {
                                    "time_ms": row_dict["time_ms"],
                                    "player_name": player_info[player_league_id]["player_name"],
                                    "league_id": player_league_id,
                                    "team": player_info[player_league_id]["team"],
                                    "x": 20+z,
                                    "y": 20,
                                    "z": 0,
                                    "match_time_seconds_column": match_time_seconds,
                                    "match_time_minutes_column": match_time_seconds / 60
                                }
                                final_reduced_list.append(missing_player)
                                # Mark this player as having had a missing entry added
                                added_missing_entries.add(player_league_id)
                                print(f"Added missing entry for league_id {player_league_id} at time {match_time_seconds}")

                # Sort the counts from highest to lowest
                sorted_player_count = dict(sorted(player_count.items(), key=lambda item: item[1], reverse=True))
                sorted_league_id_count = dict(sorted(league_id_count.items(), key=lambda item: item[1], reverse=True))

                #named_dict = {"data": final_reduced_list}

                # Get the current date
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Define the version number
                version_number = "1.0.0"

                # Add a comment with the current date and version number to the dictionary
                named_dict = {
                    "__comment__": f"This file contains reduced position data. Data generated on {current_date}. Version: {version_number}.",
                    "data": final_reduced_list,
                    "player_count": sorted_player_count,
                    "league_id_count": sorted_league_id_count
                }

                # Save the list of dictionaries as a JSON file
                with open(file_path_reduced_json, 'w') as f:
                    json.dump(named_dict, f, indent=4)
                print(f"Final reduced array saved to {file_path_reduced_json}")

                # Reduced data: time, league_id, team, x, y, z
                reduced_data_dict = arr_reduced #[:, [0, 4, 5, 6, 8, 9, 10]] #.astype({'4': str, '5': int, '6': int, '8': float, '9': float, '10': float})
                print("Reduced Data dict with names", reduced_data_dict)
                
                grouped_data = {}
                last_seen = {}
                player_info = {}

                for row in reduced_data_dict:
                    time_ms = int(row[0])
                    if time_ms not in grouped_data:
                        grouped_data[time_ms] = []
                    league_id = int(row[2])

                    # Set "z" value to 0 if league_id is greater than 5
                    z_value = float(row[6])
                    if league_id > 5:
                        z_value = 0

                    player_data = {
                        "player_name": str(row[1]),
                        "league_id": league_id,
                        "team": int(row[3]),
                        "x": float(row[4]),
                        "y": float(row[5]),
                        "z": z_value,
                        "match_time_seconds_column": int(row[7])
                    }

                    grouped_data[time_ms].append(player_data)
                    
                    # Update last_seen and player_info
                    last_seen[league_id] = player_data["match_time_seconds_column"]
                    player_info[league_id] = {
                        "player_name": player_data["player_name"],
                        "team": player_data["team"]
                    }

                # Check for missing players and add entries
                time_keys = sorted(grouped_data.keys())
                added_missing_entries = set()

                for idx, current_time in enumerate(time_keys):
                    current_second = grouped_data[current_time][0]["match_time_seconds_column"]
                    z=0
                    if current_second >= 5:
                        for player_league_id, last_seen_time in list(last_seen.items()):
                            if current_second - last_seen_time == 4 and player_league_id not in added_missing_entries:
                                z+=1
                                missing_player = {
                                    "player_name": player_info[player_league_id]["player_name"],
                                    "league_id": player_league_id,
                                    "team": player_info[player_league_id]["team"],
                                    "x": 20+z,
                                    "y": 20,
                                    "z": 0,
                                    "match_time_seconds_column": current_second
                                }
                                grouped_data[current_time].append(missing_player)
                                added_missing_entries.add(player_league_id)
                                print(f"Added missing entry for league_id {player_league_id} at time {current_time}")

                    # Update last_seen for all players in this time step
                    for player in grouped_data[current_time]:
                        last_seen[player["league_id"]] = current_second



                # Save the grouped data as JSON
                with open(file_path_grouped_json, 'w') as f:
                    json.dump(grouped_data, f, indent=4)
                print(f"Grouped data saved to {file_path_grouped_json}")

                # Initialize a new dictionary to hold the final data with vector lengths
                final_grouped_data = {}

                # List of sorted time keys
                time_keys = list(grouped_data.keys())
                #print("Time keys:", time_keys)

                # Iterate through each time entry
                for idx, current_time in enumerate(time_keys):
                    final_grouped_data[current_time] = []
                    players = grouped_data[current_time]
                    
                    # Check if there's a next time entry
                    if idx + 1 < len(time_keys):
                        next_time = time_keys[idx + 1]
                        #print("Next time:", next_time)
                        next_players = grouped_data[next_time]
                        #print(  "Next players:", next_players)
                        
                        for player in players:
                            league_id = player["league_id"]
                            #print("League ID", league_id)

                            # Set "z" value to 0 if league_id is greater than 5
                            if league_id > 5:
                                player["z"] = 0

                            x0 = player["x"]
                            #print("x0", x0)
                            y0 = player["y"]
                            #print("y0", y0  )
                            
                            # Find the next second data for the same player
                            next_player_data = next((p for p in next_players if p["league_id"] == league_id), None)
                            
                            if next_player_data:
                                x1 = next_player_data["x"]
                                #print(  "x1", x1)
                                y1 = next_player_data["y"]
                                #print(  "y1", y1)
                                length = vector_length(x0, x1, y0, y1)
                                player["vector_length_plus1"] = round(length, 3)
                            else:
                                player["vector_length_plus1"] = 0  # No matching player for next second
                            
                            final_grouped_data[current_time].append(player)
                    else:
                        # No next second available, set vector_length to None for all players
                        for player in players:
                            player["vector_length_plus1"] = 0
                            final_grouped_data[current_time].append(player)

                    # Check if there's a previous time entry
                    if idx - 1 >= 0:
                        prev_time = time_keys[idx - 1]
                        prev_players = grouped_data[prev_time]
                        
                        for player in final_grouped_data[current_time]:
                            league_id = player["league_id"]

                            # Set "z" value to 0 if league_id is greater than 5
                            if league_id > 5:
                                player["z"] = 0

                            x0 = player["x"]
                            y0 = player["y"]
                            
                            # Find the previous second data for the same player
                            prev_player_data = next((p for p in prev_players if p["league_id"] == league_id), None)
                            
                            if prev_player_data:
                                x_prev = prev_player_data["x"]
                                y_prev = prev_player_data["y"]
                                length_prev = vector_length(x0, x_prev, y0, y_prev)
                                player["vector_length_minus1"] = round(length_prev, 3)
                            else:
                                player["vector_length_minus1"] = 0  # No matching player for previous second
                    else:
                        # No previous second available, set vector_length to None for all players
                        for player in final_grouped_data[current_time]:
                            player["vector_length_minus1"] = 0

                # Calculate the average of vector_length-1 and vector_length+1
                for current_time in final_grouped_data:
                    for player in final_grouped_data[current_time]:
                        length_prev = player.get("vector_length_minus1")
                        length_next = player.get("vector_length_plus1")
                        
                        if length_prev != 0 and length_next != 0:
                            player["vector_length_avg"] = round((length_prev + length_next) / 2, 3)
                        elif length_prev == 0:
                            player["vector_length_avg"] = length_next
                        elif length_next == 0: 
                            player["vector_length_avg"] = length_prev  
                                

                # Save the final data with vector lengths as JSON
                with open(file_path_final_json, 'w') as f:
                    json.dump(final_grouped_data, f, indent=4)
                print(f"Final grouped data with vector lengths saved to {file_path_final_json}")

                # Convert the JSON string to a dictionary (if it's not already a dictionary)
                grouped_data_dict = grouped_data if isinstance(grouped_data, dict) else json.loads(grouped_data)

                # Create a lookup dictionary for vector lengths by player name and time
                vector_lengths_lookup = {}
                for time_ms, players in grouped_data_dict.items():
                    for player in players:
                        player_key = (int(time_ms), player["league_id"])
                        vector_lengths_lookup[player_key] = {
                            "vector_length_plus1": player.get("vector_length_plus1", 0),
                            "vector_length_minus1": player.get("vector_length_minus1", 0),
                            "vector_length_avg": player.get("vector_length_avg", 0)
                        }

                # Add vector lengths to each player in final_reduced_list
                for entry in final_reduced_list:
                    player_key = (entry["time_ms"], entry["league_id"])
                    if player_key in vector_lengths_lookup:
                        entry.update(vector_lengths_lookup[player_key])
                    else:
                        entry["vector_length_plus1"] = 0
                        entry["vector_length_minus1"] = 0
                        entry["vector_length_avg"] = 0
                    
                    # Berechnen der Geschwindigkeitsvariationen
                    variations = velocity_variations(0, 0, entry["vector_length_avg"], 0)
                    entry.update(variations)

                # Detect throw events
                throw_triggers = detect_throw_events(final_grouped_data)

                # Add throw triggers to the player data
                for time, players in final_grouped_data.items():
                    if time in throw_triggers:
                        for player in players:
                            player_id = player['league_id']
                            player['throw_trigger'] = throw_triggers[time].get(player_id, 0)

                # Add throw triggers to each player in final_reduced_list
                for entry in final_reduced_list:
                    time = entry['time_ms']
                    player_id = entry['league_id']
                    if time in throw_triggers:
                        entry['throw_trigger'] = throw_triggers[time].get(player_id, 0)
                    else:
                        entry['throw_trigger'] = 0

                try:
                     # Create the final structure
                    final_reduced_list_dict = {
                        "__comment__": f"This file contains reduced position data with vector lengths, throw triggers, and velocity variations. Data generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Version: 1.1.0.",
                        "data": final_reduced_list
                    }

                    logging.info(f"Performing clustering using {chosen_method} method...")
                    start_time = time_module.time()  # Use the renamed time module
                    movement_thresholds = perform_clustering_variations(final_reduced_list_dict['data'], 
                                                                        method=chosen_method, 
                                                                        manual_thresholds=manual_thresholds if use_manual else None)
                    end_time = time_module.time()  # Use the renamed time module
                    computation_time = end_time - start_time

                    # Add thresholds to the final_reduced_list_dict
                    final_reduced_list_dict.update(movement_thresholds)

                    # Add a comment to indicate which clustering method was used and how long it took
                    if use_manual:
                        final_reduced_list_dict['__threshold_comment__'] = f"Manual thresholds were used for clustering. Computation time: {computation_time:.2f} seconds."
                    else:
                        final_reduced_list_dict['__threshold_comment__'] = f"Thresholds were calculated using {chosen_method} clustering for different velocity variations. Computation time: {computation_time:.2f} seconds."

                    # Save the updated final_reduced_list as JSON
                    logging.info(f"Attempting to save file to: {file_path_final_json_vector_arr_red}")
                    with open(file_path_final_json_vector_arr_red, 'w') as f:
                        json.dump(final_reduced_list_dict, f, indent=4)
                    logging.info(f"Successfully saved updated final reduced list to {file_path_final_json_vector_arr_red}")
                    logging.info(f"Total computation time: {computation_time:.2f} seconds")

                    return final_reduced_list_dict
                except Exception as e:
                    logging.error(f"Error in process_csv: {str(e)}")
                    logging.error(traceback.format_exc())
                    raise
            

            except IndexError:
                    print("End of data")
                
            print("Reduced array:", arr_reduced)
       
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    except csv.Error as e:
        print(f"CSV error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


    with open(csv_path_playerInfo, 'r') as csvfile_PlayerInfo:
            print("Opened CSV file Player Info") 
            list_reader_2 = csv.reader(csvfile_PlayerInfo, delimiter=',')  # "," and ";" prüfen in der CSV Datei und hier anpassen
            # Get the headers (first row)
            headers_player = next(list_reader_2)
            # Print the headers
            #print("Headers Player Info:", headers_player)

            data_list2 = list(list_reader_2)
            data_list_array2 = np.array(data_list2)
            #print("List data shape", data_list_array2.shape)

            reduced_data2 = data_list_array2[:, [0, 1, 2, 3, 5, 6, 8, 9]] # ID, Last Name, First Name, Number, Team,Height, Weight
            #print("Reduced Data Player Information", reduced_data2)

            # Replace empty strings with NaN in the entire array
            reduced_data2 = np.where(reduced_data2 == '', np.nan, reduced_data2)
            #print("Reduced Data with NaN for empty strings", reduced_data2)

            # Save the array to a text file
            np.savetxt(file_path_reduced_playerInfo_txt, reduced_data2, delimiter=',', fmt='%s'), #, fmt='%s' for string
            #print(f"Array Player Info saved to {file_path_reduced_playerInfo_txt}")

            # Convert columns to appropriate types for mean calculation
            height_col = reduced_data2[:, -2].astype(float)
            weight_col = reduced_data2[:, -1].astype(float)

            # Calculate the mean for height and weight, ignoring NaNs
            mean_height = np.nanmean(height_col)
            mean_weight = np.nanmean(weight_col)

            # Fill NaN values with the mean value
            height_col = np.where(np.isnan(height_col), mean_height, height_col)
            weight_col = np.where(np.isnan(weight_col), mean_weight, weight_col)

            # Assign the filled columns back to the reduced data
            reduced_data2[:, -2] = np.round((height_col.astype(float) / 100), 2)  # Convert height to meters
            reduced_data2[:, -1] = np.round((weight_col.astype(float) / 100), 2)  # Weight as float

            print("Updated Reduced Data Player Information", reduced_data2)

            # Define the original keys and the corresponding JSON keys
            original_keys = ["ID", "Last Name", "First Name", "Number", "Position", "Team", "Height", "Weight"]
            json_keys = ["ID", "Last_Name", "First_Name", "Number", "Position", "Team", "Height", "Weight"]

            # Identify which columns contain float values
            float_columns = ["Height", "Weight"]

            # Identify the columns that contain integer values
            int_columns = ["ID", "Number"]

            # Identify the columns that contain string values
            string_columns = ["Last_Name", "First_Name", "Position", "Team"]

            # Convert the NumPy array to a list of dictionaries
            final_reduced_list2 = []
            for row in reduced_data2:
                row_dict2 = {}
                for i, key in enumerate(original_keys):
                    json_key = json_keys[i]
                    if json_key in float_columns:
                        row_dict2[json_key] = float(row[i]) #if row[i] else 0.0 # Convert to float, default to 0.0 if empty 
                        #print("Float", row_dict2[json_key], "Nummer", i)
                    elif json_key in int_columns:
                        row_dict2[json_key] = int(row[i]) #if row[i] else 0  # Convert to int, default to 0 if empty
                        #print("Int", row_dict2[json_key], "Nummer", i)
                    elif json_key in string_columns:
                        row_dict2[json_key] = translate_special_chars(str(row[i]))
                        #print("string", row_dict2[json_key], "Nummer", i)
                final_reduced_list2.append(row_dict2)

            #named_dict2 = {"PlayerInfo": final_reduced_list2}

            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Define the version number
            version_number = "1.0.1"

            # Add a comment with the current date and version number to the dictionary
            named_dict2 = {
                "__comment__": f"This file contains player information including positions. Data generated on {current_date}. Version: {version_number}.",
                "PlayerInfo": final_reduced_list2
            }

            # Save the list of dictionaries as a JSON file
            with open(file_path_reduced_json2, 'w') as f:
                json.dump(named_dict2, f, indent=4)
            print(f"Final reduced array Player Information saved to {file_path_reduced_json2}")

    with open(csv_path_eventInfo, 'r') as csvfile_eventInfo:
            print("Opened CSV file Player Info") 
            list_reader_3 = csv.reader(csvfile_eventInfo, delimiter=',')  # "," and ";" prüfen in der CSV Datei und hier anpassen
            # Get the headers (first row)
            headers_player = next(list_reader_3)
            # Print the headers
            print("Headers Player Info:", headers_player)

            data_list3 = list(list_reader_3)
            data_list_array3 = np.array(data_list3)
            print("List data shape", data_list_array3.shape)

            # Update the columns to include player and players information
            reduced_data3 = data_list_array3[:, [1, 2, 6, 8, 9]]  # ID, Type, match_clock, player, players
            print("Reduced Data Event Information", reduced_data3)

            # Convert the NumPy array to a list of dictionaries
            final_reduced_list3 = []
            row_id = 1

            for row in reduced_data3:
                row_dict3 = {
                    "row_id": row_id,
                    "id": int(row[0]),
                    "event_name": str(row[1]),
                    "match_clock": row[2],
                    "match_clock_seconds": convert_time_to_seconds(row[2]),
                    "match_clock_minutes": convert_time_to_minutes(row[2]),
                    "players": []
                }

                player_data = parse_player(row[3])
                players_data = parse_players(row[4])

                all_players = players_data if players_data else ([player_data] if player_data else [])

                for i, player in enumerate(all_players):
                    player_info = {
                        "player_id": extract_player_id(player.get('id', '')),
                        "player_name": reformat_player_name(player.get('name', '')),
                        "player_type": player.get('type', ''),
                        "is_attacker": (i == 0)
                    }
                    row_dict3["players"].append(player_info)

                final_reduced_list3.append(row_dict3)
                row_id += 1

            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Define the version number
            version_number = "1.0.2"

            # Add a comment with the current date and version number to the dictionary
            named_dict3 = {
                "__comment__": f"This file contains event information optimized for Unity with integer player IDs and reformatted names. Data generated on {current_date}. Version: {version_number}.",
                "EventInfo": final_reduced_list3
            }

            # Save the list of dictionaries as a JSON file
            with open(file_path_reduced_json3, 'w') as f:
                json.dump(named_dict3, f, indent=4)
            print(f"Final reduced array Event Information saved to {file_path_reduced_json3}")


    # Create a dictionary to store the grouped data
    grouped_data_event = {}
    grouped_data_bySeconds = {}

    startSeconds_beforeEvent = 5
    endSeconds_afterEvent = 10

    # Initialize the event_key counter
    event_key = 1

    # Iterate through each event in final_reduced_list3
    for event in final_reduced_list3:
        row_id = event["row_id"]
        match_clock_seconds = event["match_clock_seconds"]
        
        # Initialize the list for matching rows
        matching_rows_range = []
        matching_rows_exact = []
        
        # Iterate through the grouped data dictionary
        for time_ms, rows in grouped_data.items():
            for row in rows:
                # Filter based on the time range condition
                if match_clock_seconds - startSeconds_beforeEvent <= row["match_time_seconds_column"] <= match_clock_seconds + endSeconds_afterEvent:
                    matching_rows_range.append(row)
                # Add rows where the match time seconds match exactly
                if row["match_time_seconds_column"] == match_clock_seconds:
                    matching_rows_exact.append(row)

        # Combine the two lists and remove duplicates
        combined_matching_rows = {json.dumps(row): row for row in (matching_rows_range + matching_rows_exact)}
        combined_matching_rows = list(combined_matching_rows.values())

        # Add event_key to each item in combined_matching_rows
        unique_combined_matching_rows = []
        for item in combined_matching_rows:
            item_with_key = item.copy()
            item_with_key["event_key"] = event_key
            unique_combined_matching_rows.append(item_with_key)
        event_key += 1  # Increment the event_key for uniqueness
        
        # Add the combined matched values to the dictionary with row_id as the key
        grouped_data_event[row_id] = unique_combined_matching_rows
        
            # Group by match_time_seconds_column within the event data in grouped_data_bySeconds
        if row_id not in grouped_data_bySeconds:
            grouped_data_bySeconds[row_id] = {}
        for row in unique_combined_matching_rows:
            match_time = row["match_time_seconds_column"]
            if match_time not in grouped_data_bySeconds[row_id]:
                grouped_data_bySeconds[row_id][match_time] = []
            grouped_data_bySeconds[row_id][match_time].append(row)

    # Initialize dictionaries to keep track of player information and last seen time
    last_seen = {}
    player_info = {}
    added_missing_entries = set()

    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Define the version number
    version_number = "1.0.2"  # Updated version number

    # Restructure the data to match the desired format
    final_output = {
        "__comment__": f"This file contains event and positional information with added missing player entries and velocity variations. Data generated on {current_date}. Version: {version_number}.",
        "event_data": []
    }

    # Process and add data to final_output
    for event_key, event_data in grouped_data_event.items():
        event_data_sorted = sorted(event_data, key=lambda x: x['match_time_seconds_column'])
        processed_event_data = []
        
        for idx, player_data in enumerate(event_data_sorted):
            match_time_seconds = player_data['match_time_seconds_column']
            league_id = player_data['league_id']
            
            # Update last_seen and player_info
            last_seen[league_id] = idx
            player_info[league_id] = {
                "player_name": player_data['player_name'],
                "team": player_data['team']
            }
            
            z = 0
            # Check for players missing for 4 consecutive entries
            if idx >= 3:
                for missing_league_id, last_idx in list(last_seen.items()):
                    if idx - last_idx == 4 and missing_league_id not in added_missing_entries:
                        # Player has been missing for 4 entries and we haven't added an entry yet
                        z += 1
                        missing_player = {
                            "player_name": player_info[missing_league_id]["player_name"],
                            "league_id": missing_league_id,
                            "team": player_info[missing_league_id]["team"],
                            "x": 20+z,
                            "y": 20,
                            "z": 0,
                            "match_time_seconds_column": match_time_seconds,
                            "vector_length_plus1": 0,
                            "vector_length_minus1": 0,
                            "vector_length_avg": 0,
                            "v": 0,
                            "log_v": 0,
                            "sqrt_v": 0,
                            "throw_trigger": 0,
                            "event_key": player_data['event_key']
                        }
                        processed_event_data.append(missing_player)
                        added_missing_entries.add(missing_league_id)
                        print(f"Added missing entry for league_id {missing_league_id} at time {match_time_seconds} for event {event_key}")

            # Calculate velocity variations
            v = player_data.get('vector_length_avg', 0)
            log_v = np.log(v + 1)  # +1 to avoid log(0)
            sqrt_v = np.sqrt(v)

            # Add the current entry to the processed list with reordered fields and velocity variations
            processed_player_data = {
                "player_name": player_data['player_name'],
                "league_id": player_data['league_id'],
                "team": player_data['team'],
                "x": player_data['x'],
                "y": player_data['y'],
                "z": player_data['z'],
                "match_time_seconds_column": player_data['match_time_seconds_column'],
                "vector_length_plus1": player_data.get('vector_length_plus1', 0),
                "vector_length_minus1": player_data.get('vector_length_minus1', 0),
                "vector_length_avg": player_data.get('vector_length_avg', 0),
                "v": round(v, 3),
                "log_v": round(log_v, 16),  # Increased precision for log_v
                "sqrt_v": round(sqrt_v, 16),  # Increased precision for sqrt_v
                "throw_trigger": player_data.get('throw_trigger', 0),
                "event_key": player_data['event_key']
            }
            processed_event_data.append(processed_player_data)
        
        # Add processed event data to final_output
        final_output["event_data"].extend(processed_event_data)

    # Reset added_missing_entries for potential use in future processing
    added_missing_entries.clear()

    # Save the final_output as JSON
    with open(file_path_grouped_json_event, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"Grouped data with missing player entries and velocity variations saved to {file_path_grouped_json_event}")


    # Third output structure: Keep each "match_time_seconds_column" as a key in "grouped_data_bySeconds" dictionary 
    final_output_by_match_time = {
        "__comment__": f"This file contains event and positional information grouped by match time seconds. Data generated on {current_date}. Version: {version_number}.",
        "event_data": {}  # Initialize as a dictionary
    }

        # Move all events into the event_data dictionary grouped by match_time_seconds_column in final_output_by_match_time
    for row_id, match_time_dict in grouped_data_bySeconds.items():
        final_output_by_match_time["event_data"][str(row_id)] = {}
        for match_time, data in match_time_dict.items():
            final_output_by_match_time["event_data"][str(row_id)][str(match_time)] = data

    with open(file_path_grouped_json_by_match_time, 'w') as f:
        json.dump(final_output_by_match_time, f, indent=4)

    print(f"Grouped data by match time saved to {file_path_grouped_json_by_match_time}")

    # Create a copy of final_output_by_match_time
    final_output_copy = final_output_by_match_time.copy()
    # Create the EventKeyData folder if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Iterate through each event_key in the event_data
    for event_key, match_time_data in final_output_by_match_time["event_data"].items():
        # Create a new dictionary for this event
        new_output = {
            "__comment__": final_output_by_match_time["__comment__"],
            "event_data": {
                event_key: {}
            }
        }
        
        # Process each match time and its corresponding data
        for match_time, player_data_list in match_time_data.items():
            new_output["event_data"][event_key][match_time] = []
            
            for player_data in player_data_list:
                # Calculate velocity variations
                v = player_data.get('vector_length_avg', 0)
                log_v = np.log(v + 1)  # +1 to avoid log(0)
                sqrt_v = np.sqrt(v)
                
                # Create updated player data with velocity variations and in the desired order
                updated_player_data = OrderedDict([
                    ("player_name", player_data.get("player_name")),
                    ("league_id", player_data.get("league_id")),
                    ("team", player_data.get("team")),
                    ("x", player_data.get("x")),
                    ("y", player_data.get("y")),
                    ("z", player_data.get("z")),
                    ("match_time_seconds_column", player_data.get("match_time_seconds_column")),
                    ("vector_length_plus1", player_data.get("vector_length_plus1")),
                    ("vector_length_minus1", player_data.get("vector_length_minus1")),
                    ("vector_length_avg", player_data.get("vector_length_avg")),
                    ("v", round(v, 3)),
                    ("log_v", round(log_v, 16)),  # Increased precision for log_v
                    ("sqrt_v", round(sqrt_v, 16)),  # Increased precision for sqrt_v
                    ("throw_trigger", player_data.get("throw_trigger")),
                    ("event_key", player_data.get("event_key"))
                ])
                
                new_output["event_data"][event_key][match_time].append(updated_player_data)
            
        # Create the file name
        file_name = f"event_{event_key}.json"
        file_path = os.path.join(base_path, file_name)
        
        # Save the new dictionary as a JSON file
        with open(file_path, 'w') as f:
            json.dump(new_output, f, indent=4)
        
        print(f"Created {file_name}")

# Add these new functions at the end of your script
def parse_player(player_str):
    if not player_str:
        return {}
    try:
        return ast.literal_eval(player_str)
    except:
        return {}

def parse_players(players_str):
    if not players_str:
        return []
    try:
        return ast.literal_eval(players_str)
    except:
        return []

def extract_player_id(id_str):
    try:
        return int(id_str.split(':')[-1])
    except:
        return 0  # Return 0 or some default value if parsing fails

def reformat_player_name(name):
    parts = name.split(', ')
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return name  # Return original name if it doesn't match expected format

def detect_throw_events(data, proximity_threshold=0.8):
    throw_triggers = {}
    
    # Sort the data by match_time_seconds_column
    sorted_times = sorted(data.keys(), key=lambda x: data[x][0]['match_time_seconds_column'])
    
    for time in sorted_times:
        frame_data = data[time]
        current_second = frame_data[0]['match_time_seconds_column']
        
        ball_positions = {}
        player_positions = {}
        throw_triggers[time] = {}

        # Separate ball and player positions
        for entity in frame_data:
            if 1 <= entity['league_id'] <= 3:  # Ball
                ball_positions[entity['league_id']] = (entity['x'], entity['y'])
                throw_triggers[time][entity['league_id']] = 0  # Ball always has throw_trigger = 0
            else:  # Player
                player_positions[entity['league_id']] = (entity['x'], entity['y'])
                throw_triggers[time][entity['league_id']] = 0  # Initialize player throw_trigger to 0

        # Check for throw events
        for ball_id, ball_pos in ball_positions.items():
            nearest_player = None
            nearest_distance = float('inf')
            
            for player_id, player_pos in player_positions.items():
                distance = ((player_pos[0] - ball_pos[0])**2 + (player_pos[1] - ball_pos[1])**2)**0.5
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_player = player_id

            if nearest_player and nearest_distance < proximity_threshold:
                throw_triggers[time][nearest_player] = 1  # Set throw trigger for nearest player

    return throw_triggers

def process_jersey_colors(csv_path, json_output_path):
    jersey_data = {}
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        headers = next(reader)  # Skip the header row
        
        for row in reader:
            team = row[0]
            jersey_data[team] = {
                "Home": {
                    "Goalie": {
                        "Shirt": row[1],
                        "Shorts": row[2],
                        "Shoes": row[3]
                    },
                    "FieldPlayer": {
                        "Shirt": row[4],
                        "Shorts": row[5],
                        "Shoes": row[6]
                    }
                },
                "Away": {
                    "Goalie": {
                        "Shirt": row[7],
                        "Shorts": row[8],
                        "Shoes": row[9]
                    },
                    "FieldPlayer": {
                        "Shirt": row[10],
                        "Shorts": row[11],
                        "Shoes": row[12]
                    }
                }
            }
    
    with open(json_output_path, 'w') as jsonfile:
        json.dump(jersey_data, jsonfile, indent=4)

    print(f"Jersey color data saved to {json_output_path}")

def process_event_files(input_base_path, output_base_path):
    # Create the EventData folder if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)

    # Iterate through each file in the input directory
    for filename in os.listdir(input_base_path):
        if filename.startswith("event_") and filename.endswith(".json"):
            file_path = os.path.join(input_base_path, filename)
            
            # Read the original file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create a new dictionary for this event with simplified structure
            new_output = {
                "__comment__": data["__comment__"],
                "event_data": []
            }
            
            # Process the event data
            for event_key, match_time_data in data["event_data"].items():
                for match_time, player_data in match_time_data.items():
                    new_output["event_data"].extend(player_data)
            
            # Create the new file name
            new_filename = filename
            new_file_path = os.path.join(output_base_path, new_filename)
            
            # Save the new dictionary as a JSON file
            with open(new_file_path, 'w') as f:
                json.dump(new_output, f, indent=4)
            
            print(f"Created {new_filename} in EventData folder")

if __name__ == "__main__":
    # Definieren Sie manuelle Schwellwerte (passen Sie diese Werte nach Bedarf an)
    manual_thresholds = {
        'v': {
            'idle_threshold': 0.1,
            'walk_threshold': 0.22,
            'run_threshold': 5.0
        },
        'log_v': {
            'idle_threshold': 0.05,
            'walk_threshold': 0.15,
            'run_threshold': 1.5
        },
        'sqrt_v': {
            'idle_threshold': 0.3,
            'walk_threshold': 0.5,
            'run_threshold': 2.0
        }
    }

    csv_path = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\TBVLemgoLippe_SCMagdeburg_sr_sport_event_42307971_position_data_only_gameclock.csv"
    csv_path_playerInfo = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\2024-05-04_19-24-39_Player_information.csv"
    csv_path_eventInfo = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\2024-05-05_TBVLemgoLippe_SCMagdeburg_sr_sport_event_event_data.csv"
    csv_path_jersey = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\Jerseycolour.csv"
    file_path_reduced_pkl = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\arr_reduced.pkl"
    file_path_reduced_txt = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\arr_reduced.txt"
    file_path_reduced_json = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\array_reduced.json"
    file_path_reduced_json2 = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\array_reduced_playerInfo.json"
    file_path_reduced_json3 = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\array_reduced_eventInfo.json"
    file_path_final_json = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\final_grouped_data_with_vector_lengths.json"
    file_path_final_json_vector_arr_red = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\updated_final_reduced_list.json"
    file_path_reduced_playerInfo_txt = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\array_reduced_playerInfo.txt"
    file_path_reduced_eventInfo_txt = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\array_reduced_eventInfo.txt"
    file_path_index_csv = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\index_array.csv"
    file_path_index_json = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\index_array.json"
    file_path_grouped_json = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\grouped_data.json"
    file_path_grouped_json_event = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\grouped_data_event.json"
    file_path_grouped_json_event_structured = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\grouped_data_event_structured.json"
    file_path_grouped_json_by_match_time = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\grouped_data_by_match_time.json"
    base_path = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\EventKeyData"
    input_base_path = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\EventKeyData"
    output_base_path = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\EventData"
    json_output_path = "C:\\Users\\maxim\\OneDrive\\Desktop\\BioMechatronik\\3. Semester\\1. Masterarbeit\\Daten\\Test\\PreProcessed\\jersey_colors.json"
   
     # Prompt user for threshold method
    use_manual = input("Use manual thresholds? (y/n): ").lower().strip() == 'y'
    # Prompt user for clustering method
    print("Choose a clustering method:")
    print("1. K-means")
    print("2. Spectral Clustering")
    print("3. Gaussian Mixture Models (GMM)")
    print("4. Agglomerative Clustering")
    print("5. DBSCAN")
    print("6. Create and save clustering plots")
    
    method_choice = input("Enter the number of your choice (1-6): ").strip()
    
    method_map = {
        '1': 'kmeans',
        '2': 'spectral',
        '3': 'gmm',
        '4': 'agglomerative',
        '5': 'dbscan'
    }

    chosen_method = method_map.get(method_choice, 'kmeans')  # Default to 'kmeans' if invalid choice

    if chosen_method == 'spectral':
        print("Warning: Spectral clustering may take a long time for large datasets.")
        print("If it doesn't complete in a reasonable time, it will fall back to K-means.")
        proceed = input("Do you want to proceed? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Defaulting to K-means.")
            chosen_method = 'kmeans'
    elif chosen_method == 'dbscan':
        print("Note: DBSCAN may not always produce exactly 3 clusters.")
        print("If it can't identify 3 distinct clusters, it will fall back to K-means.")
        proceed = input("Do you want to proceed? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Defaulting to K-means.")
            chosen_method = 'kmeans'
    if chosen_method == 'gmm':
        print("Note: Gaussian Mixture Models (GMM) will be used for clustering.")
        print("This method assumes the data follows a Gaussian distribution.")
        proceed = input("Do you want to proceed? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Defaulting to K-means.")
            chosen_method = 'kmeans'

    if method_choice == '6':
        plots_dir = os.path.join(base_path, "ClusteringPlots", datetime.now().strftime("%Y%m%d_%H%M%S"))   
        try:
            # Create plots directory if it doesn't exist
            os.makedirs(plots_dir, exist_ok=True)
            
            # Ask for the data type to plot
            print("\nChoose data type to plot:")
            print("1. Regular velocity (v)")
            print("2. Log velocity (log_v)")
            print("3. Square root velocity (sqrt_v)")
            data_choice = input("Enter your choice (1-3): ").strip()
            
            # Get velocity type
            velocity_type = 'v'  # default
            if data_choice == '2':
                velocity_type = 'log_v'
            elif data_choice == '3':
                velocity_type = 'sqrt_v'
            
            # Create data for clustering
            data_dict = []
            
            # Load the processed data
            if not os.path.exists(file_path_final_json_vector_arr_red):
                print(f"Error: File not found at {file_path_final_json_vector_arr_red}")
                raise FileNotFoundError
                
            with open(file_path_final_json_vector_arr_red, 'r') as f:
                json_data = json.load(f)
            
            # Extract the chosen velocity type data
            velocities = [item[velocity_type] for item in json_data['data'] if velocity_type in item]
            
            # Create input data format expected by clustering functions
            for v in velocities:
                data_dict.append({
                    'v': v if velocity_type == 'v' else np.exp(v) - 1 if velocity_type == 'log_v' else v**2,
                    'log_v': np.log(v + 1) if velocity_type == 'v' else v if velocity_type == 'log_v' else np.log(v**2 + 1),
                    'sqrt_v': np.sqrt(v) if velocity_type == 'v' else np.sqrt(np.exp(v) - 1) if velocity_type == 'log_v' else v
                })
            
            print(f"\nCreating clustering visualizations for {velocity_type}...")
            print(f"Plots will be saved to: {plots_dir}")
            
            # Get results for all methods
            methods = ['kmeans', 'spectral', 'gmm', 'agglomerative', 'dbscan']
            all_results = {}
            
            for method in methods:
                try:
                    results = perform_clustering_variations(data_dict, method=method)
                    if results and f'cluster_centers_{velocity_type}' in results:
                        all_results[method] = results[f'cluster_centers_{velocity_type}']
                except Exception as e:
                    print(f"Error with {method}: {str(e)}")
            
            # Visualize the results
            visualize_all_clustering_results(velocities, velocity_type=velocity_type, 
                                          save_dir=plots_dir, clustering_results=all_results)
            
            print(f"Visualizations complete. Check the directory:")
            print(plots_dir)
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            print(traceback.format_exc())
    else:
        chosen_method = method_map.get(method_choice, 'kmeans')

    try:
        # Call process_csv with the new parameters
        final_reduced_list_dict = process_csv(csv_path, csv_path_playerInfo, 
                    csv_path_eventInfo, 
                    file_path_reduced_pkl, 
                    file_path_reduced_txt, 
                    file_path_index_csv, 
                    file_path_index_json, 
                    file_path_reduced_json, 
                    file_path_reduced_json2, 
                    file_path_reduced_json3, 
                    file_path_grouped_json, 
                    file_path_reduced_playerInfo_txt, 
                    file_path_reduced_eventInfo_txt, 
                    file_path_grouped_json_event, 
                    file_path_grouped_json_event_structured, 
                    file_path_grouped_json_by_match_time, 
                    base_path, file_path_final_json, 
                    file_path_final_json_vector_arr_red,
                    chosen_method=chosen_method,
                    use_manual=use_manual,
                    manual_thresholds=manual_thresholds)

        process_jersey_colors(csv_path_jersey, json_output_path)

        # Process event files to remove specified keys and save in new location
        process_event_files(input_base_path, output_base_path)
        
        print("Data pre processing succesfully finished")
        print("Next: Import data files to Unity")

        # Add visualization
        print("Creating clustering visualizations...")
        velocities = [item['v'] for item in final_reduced_list_dict['data'] if 'v' in item]
        visualize_all_clustering_results(velocities, save_dir='clustering_plots')
        print("Visualizations complete. Check the 'clustering_plots' directory.")
        
        print("Data pre processing successfully finished")
        print("Next: Import data files to Unity")


    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        logging.error(traceback.format_exc())
        print("An error occurred. Please check the log for details.")