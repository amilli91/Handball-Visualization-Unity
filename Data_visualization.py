import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import os
from datetime import datetime
import time

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def create_anonymized_mappings(df):
    #Create mappings for teams and players to anonymized names
    # Create team mapping - filter out None/NaN values and convert to list
    unique_teams = [team for team in df['team'].unique() if pd.notna(team)]
    unique_teams.sort()  # Sort in place after filtering
    team_mapping = {team: f"Team {i+1}" for i, team in enumerate(unique_teams)}
    
    # Create player mapping
    player_mapping = {}
    for team in unique_teams:
        # Get players for this team, filtering out None/NaN values
        team_players = [player for player in df[df['team'] == team]['player_name'].unique() 
                       if pd.notna(player)]
        team_players.sort()  # Sort player names
        team_num = team_mapping[team].split()[1]
        for i, player in enumerate(team_players):
            player_mapping[player] = f"Player {team_num}.{i+1}"
    
    # Handle any players with None/NaN team values
    unassigned_players = [player for player in df['player_name'].unique() 
                         if pd.notna(player) and player not in player_mapping]
    if unassigned_players:
        for i, player in enumerate(sorted(unassigned_players)):
            player_mapping[player] = f"Player 0.{i+1}"  # Use team 0 for unassigned players
    
    return team_mapping, player_mapping

def process_data(position_data, player_info):
    df = pd.DataFrame(position_data['data'])
    
    player_info_dict = {str(player['ID']): {'Position': player['Position'], 'Team': player['Team']} 
                        for player in player_info['PlayerInfo']}
    
    df['position'] = df['league_id'].astype(str).map(lambda x: player_info_dict.get(x, {}).get('Position'))
    df['team'] = df['league_id'].astype(str).map(lambda x: player_info_dict.get(x, {}).get('Team'))
    
    # Create anonymized mappings
    team_mapping, player_mapping = create_anonymized_mappings(df)
    
    # Apply mappings
    df['team'] = df['team'].map(team_mapping)
    df['player_name'] = df['player_name'].map(player_mapping)
    
    print("Unique positions:", df['position'].unique())
    print("Unique teams:", df['team'].unique())
    
    return df

def draw_court(ax):
    # [Court drawing code remains unchanged]
    court_length, court_width = 40, 20
    Torpfosten = 1.5
    Torweite = 3
    Torraum = 7
    Freiwurflinie = 11

    ax.add_patch(plt.Rectangle((0, 0), court_length, court_width, fill=False, color='black', zorder=2))
    ax.add_patch(plt.Rectangle((0, 8.5), Torpfosten, Torweite, fill=False, color='black', zorder=2))
    ax.add_patch(plt.Rectangle((38.5, 8.5), Torpfosten, Torweite, fill=False, color='black', zorder=2))
    ax.add_patch(Circle((0, 10), Torraum, fill=False, color='black', zorder=2))
    ax.add_patch(Circle((0, 10), Freiwurflinie, fill=False, color='black', zorder=2))
    ax.add_patch(Circle((40, 10), Torraum, fill=False, color='black', zorder=2))
    ax.add_patch(Circle((40, 10), Freiwurflinie, fill=False, color='black', zorder=2))
    ax.add_line(Line2D([20, 20], [0, 20], color='black', zorder=2))
    
    ax.set_aspect('equal')
    ax.set_xlabel('x position in meters')
    ax.set_ylabel('y position in meters')

def generate_filename(title):
    safe_title = "".join([c if c.isalnum() else "_" for c in title])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_title}_{timestamp}.svg"

def plot_all_players(df, title, team=None, output_dir="plots"):
    print(f"Starting plot: {title}")
    start_time = time.time()
    
    fig, ax = plt.subplots(figsize=(14, 10))
       
    if team:
        df_plot = df[df['team'] == team]
    else:
        df_plot = df
    
    print(f"Plotting for team: {team}")
    print(f"Number of data points: {len(df_plot)}")
    print(f"Unique positions in this plot: {df_plot['position'].unique()}")

    # Group players by position and sort alphabetically
    grouped_players = df_plot.groupby('position')['player_name'].unique().sort_index()
    
    # Create a color map for positions
    positions = grouped_players.index
    colors = plt.cm.rainbow(np.linspace(0, 1, len(positions)))
    color_dict = dict(zip(positions, colors))

    # Plot data and prepare legend entries
    legend_elements = []
    for position, players in grouped_players.items():
        for player in players:
            player_data = df_plot[df_plot['player_name'] == player]
            ax.scatter(player_data['x'], player_data['y'], 
                       c=[color_dict[position]], alpha=0.7, s=10)
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color_dict[position], markersize=8, 
                                              label=f"{player} ({position})"))

    draw_court(ax)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 20)
    ax.set_title(title)
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
     
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = generate_filename(title)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved as {filepath}")
    else:
        plt.show()
    
    print(f"Plot completed in {time.time() - start_time:.2f} seconds")

def plot_unique_positions(df, title, team=None, output_dir="plots"):
    print(f"Starting plot: {title}")
    start_time = time.time()
    
    fig, ax = plt.subplots(figsize=(14, 10))
       
    if team:
        df_plot = df[df['team'] == team]
    else:
        df_plot = df
    
    # Select one player for each unique position
    unique_players = df_plot.groupby('position').first().reset_index()
    
    positions = unique_players['position'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(positions)))
    color_dict = dict(zip(positions, colors))

    for _, row in unique_players.iterrows():
        position = row['position']
        player_name = row['player_name']
        player_data = df_plot[(df_plot['position'] == position) & (df_plot['player_name'] == player_name)]
        
        ax.scatter(player_data['x'], player_data['y'], 
                   c=[color_dict[position]], label=f"{player_name} ({position})", alpha=0.7, s=10)

    draw_court(ax)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 20)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    plt.tight_layout()
     
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = generate_filename(title)
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved as {filepath}")
    else:
        plt.show()
    
    print(f"Plot completed in {time.time() - start_time:.2f} seconds")

def plot_team_statistics(df, team, output_dir="plots"):
    team_data = df[df['team'] == team]
    
    metrics = [
        ('Average Velocity', 'v', 'm/s'),
        ('Number of Throws', 'throw_trigger', 'throws'),
        ('Total Distance Covered', 'vector_length_avg', 'm'),
        ('Total Amount of Positional Data', 'player_name', 'data points')
    ]

    for title, column, unit in metrics:
        plt.figure(figsize=(12, 6))
        
        if column == 'player_name':
            data = team_data.groupby('player_name').size()
        elif column == 'throw_trigger':
            data = team_data.groupby('player_name')['throw_trigger'].sum()
        elif column == 'vector_length_avg':
            data = team_data.groupby('player_name')['vector_length_avg'].sum()
        else:
            data = team_data.groupby('player_name')[column].mean()

        positions = team_data.groupby('player_name')['position'].first()
        
        data = data.replace('NONE', np.nan).astype(float)
        data = data.dropna()
        data = data.sort_values(ascending=False)

        bars = plt.bar(data.index, data.values)

        plt.title(f'{team} - {title}')
        plt.xlabel('Players')
        plt.ylabel(f'{title} ({unit})')
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

        ax = plt.gca()
        ax.set_xticklabels([f'{player}\n({positions[player]})' for player in data.index])

        plt.tight_layout()

        filename = generate_filename(f'{team}_{title.replace(" ", "_")}')
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        plt.close()
        print(f"Plot saved as {filepath}")

def analyze_data(df):
    print("Data Analysis:")
    print(f"Total number of data points: {len(df)}")
    print(f"Number of unique players: {df['player_name'].nunique()}")
    print("\nPlayer Movement Summary:")
    
    for player in df['player_name'].unique():
        player_data = df[df['player_name'] == player]
        total_distance = player_data['vector_length_avg'].sum()
        avg_velocity = player_data['v'].mean()
        position = player_data['position'].iloc[0]
        team = player_data['team'].iloc[0]
        
        print(f"\n{player} ({position}, {team}):")
        print(f"  Total distance covered: {total_distance:.2f} units")
        print(f"  Average velocity: {avg_velocity:.2f} units/s")
        print(f"  Number of throws: {player_data['throw_trigger'].sum()}")

def main(position_file_path, player_info_file_path):
    print("Loading data...")
    position_data = load_data(position_file_path)
    player_info = load_data(player_info_file_path)
    
    print("Processing data...")
    df = process_data(position_data, player_info)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "Plots")
    
    print("Plotting all players...")
    plot_all_players(df, 'All Players Positions on Handball Court', output_dir=output_dir)
    
    teams = df['team'].unique()
    for team in teams:
        print(f"\nPlotting all positions for team: {team}")
        plot_all_players(df, f'{team} Players Positions on Handball Court', team, output_dir=output_dir)
    
    for team in teams:
        print(f"\nPlotting unique positions for team: {team}")
        plot_unique_positions(df, f'{team} Unique Position Players on Handball Court', team, output_dir=output_dir)
    
    for team in teams:
        print(f"\nCreating statistical plots for team: {team}")
        plot_team_statistics(df, team, output_dir=output_dir)
    
    print("\nAnalyzing data...")
    analyze_data(df)

if __name__ == "__main__":
    base_path = os.path.join(
        "C:", os.sep, "Users", "maxim", "OneDrive", "Desktop", "BioMechatronik", 
        "3. Semester", "1. Masterarbeit", "Daten", "Test", "PreProcessed"
    )
    
    position_file_path = os.path.join(base_path, "updated_final_reduced_list.json")
    player_info_file_path = os.path.join(base_path, "array_reduced_playerInfo.json")
    
    if not os.path.exists(position_file_path):
        print(f"Error: File not found at {position_file_path}")
    elif not os.path.exists(player_info_file_path):
        print(f"Error: File not found at {player_info_file_path}")
    else:
        main(position_file_path, player_info_file_path)