using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using Newtonsoft.Json;
using UnityEngine.UI;
//using Newtonsoft.Json.Linq;
using System.IO;
using System;

public class DataReader3 : MonoBehaviour
{
    // JSON files for both data sets
    public TextAsset jsonFile_array_reduced;
    public TextAsset jsonFile_index_array;
    public TextAsset jsonFile_player_info; // New JSON file for player info
    public TextAsset jsonFile_event_info; // New JSON file for event info
    public TextAsset jsonFile_event_positional_info; // New JSON file for event and positonal info
    public GameObject arrowPrefab; // Arrow prefab to be assigned in the Inspector
    public GameObject textPrefab; // Text prefab to be assigned in the Inspector
    public Material transparentMaterial; // Material with transparency for out-of-bounds objects
    private Dictionary<int, float> lastMovedTimes; // Track last moved time for each player

    private LineRenderer lineRenderer;

    public TextMeshProUGUI team1PlayerCountText; // Reference to the TextMeshProUGUI component for Team 1
    public TextMeshProUGUI team2PlayerCountText; // Reference to the TextMeshProUGUI component for Team 2
    private int team1PlayerCount = 0;
    private int team2PlayerCount = 0;

    public int playMatch = 0; // Variable to control the match playback
    public int playAllEvents = 0;
    public int chooseEvent = 0;
    public int selectedBallIndex = 0; // 0 for nearest, 1 for Ball1, 2 for Ball2, 3 for Ball3
    public bool showBallArrow = false; // Show or hide the arrow for the selected ball
    public bool showPlayerArrows = true; // Add this near the top of your class
    public bool useVectorLengthMinus1 = false; // Use vector_length_minus1 instead of vector_length_plus1
    private bool lastShowBallArrow;
    private bool lastShowPlayerArrows;

    //speed of interpolation
    public float gameSpeed = 0; //A lower value will result in slower, smoother movement:

    // how frequently new positions are set
    public float eventSpeed = 0; // Adjust this value in the Inspector

    public class Billboard : MonoBehaviour
{
    void LateUpdate()
    {
        transform.LookAt(Camera.main.transform);
        transform.Rotate(0, 180, 0);
    }
}

     [Tooltip("Select the player info display mode")]
    public PlayerInfoDisplayMode playerInfoDisplayMode = PlayerInfoDisplayMode.NumberOnly;

    public TextAsset jerseyColorsJson; // Assign this in the Inspector
    private JerseyColors jerseyColors;
    private Dictionary<int, string> teamIdToName; // Add this line
    private Dictionary<int, string> playerPositions;
    private Dictionary<int, string> leagueIdToTeamMap;
    private Dictionary<string, TeamColors> teamColorData;

    // Class for the first JSON data set
    [System.Serializable]
    public class ReducedData
    {
        public int time_ms;
        public string player_name;
        public int league_id;
        public int team;
        public float x;
        public float y;
        public float z;
        public int match_time_seconds_column;
        public float match_time_minutes_column;
        public float vector_length_plus1;
        public float vector_length_minus1;
        public float vector_length_avg;
        public float v;
    public float log_v;
    public float sqrt_v;
        public int throw_trigger;
    }

    [System.Serializable]
    public class DataList
    {
        public ReducedData[] data; // Ensure this matches the JSON file key name on top
    }

    public DataList myDataList;

    // Class for the second JSON data set
    [System.Serializable]
    public class UniqueId
    {
        public int league_id;
        public int team;
        public int index;
    }

    [System.Serializable]
    public class UniqueIdsList
    {
        public UniqueId[] league_teams; // Make sure this matches the JSON file key name on top
    }

    public UniqueIdsList myUniqueIdsList;

    // Class for the player info JSON data set
    [System.Serializable]
    public class PlayerInfo
    {
        public int ID;
        public string Last_Name;
        public string First_Name;
        public int Number;
        public string Position;
        public string Team;
        public float Height;
        public float Weight;
    }

    [System.Serializable]
    public class PlayerInfoList
    {
        public PlayerInfo[] PlayerInfo; // Ensure this matches the JSON file key name on top
    }

    public PlayerInfoList myPlayerInfoList;

    // Class for the event info JSON data set
    
    [System.Serializable]
    public class PlayersEvent
    {
        public int player_id;
        public string player_name;
        public string player_type;
        public bool is_attacker;
    }

    [System.Serializable]
    public class EventInfo
    {
        public int row_id;
        public int id;
        public string event_name;
        public string match_clock;
        public int match_clock_seconds;
        public float match_clock_minutes;
        public List<PlayersEvent> players;
    }

    [System.Serializable]
    public class EventInfoList
    {
        public string __comment__;
        public List<EventInfo> EventInfo;
    }

    public EventInfoList myEventInfoList;




    // Class for the event and positional info JSON data set
    [System.Serializable]
    public class EventPositionalData
    {
        public string player_name;
        public int league_id;
        public int team;
        public float x;
        public float y;
        public float z;
        public int match_time_seconds_column;
        public float vector_length_plus1; // Note: JSON uses "vector_length+1", but C# doesn't allow '+' in variable names
        public float vector_length_minus1; // Note: JSON uses "vector_length-1"
        public float vector_length_avg;
        public int event_key;
        public float v;
    public float log_v;
    public float sqrt_v;
        public int throw_trigger;
    }

    [System.Serializable]
    public class EventPositionalDataList
    {
        public EventPositionalData[] event_data; // Ensure this matches the JSON file key name on top
    }

    public EventPositionalDataList myEventPositionalDataList;


    [System.Serializable]
   public class RootObject
    {
        public string __comment__;
        public Dictionary<string, Dictionary<string, List<EventPositionalData>>> event_data;
    }

    [System.Serializable]
    public class ClusterCenters
    {
        public float idle_threshold;
        public float walk_threshold;
        public float run_threshold;
    }

    [System.Serializable]
public class AllClusterCenters
{
    public ClusterCenters cluster_centers_v;
    public ClusterCenters cluster_centers_log_v;
    public ClusterCenters cluster_centers_sqrt_v;
}

[System.Serializable]
public class JsonData
{
    public ReducedData[] data;
    public ClusterCenters cluster_centers_v;
    public ClusterCenters cluster_centers_log_v;
    public ClusterCenters cluster_centers_sqrt_v;
    public string __threshold_comment__;
}

    [System.Serializable]
    public class JerseyColors : Dictionary<string, TeamColors>
    {
        // This class is now a Dictionary<string, TeamColors>
    }

    [System.Serializable]
    public class TeamColors
    {
        public PlayerColors Home;
        public PlayerColors Away;
    }

    [System.Serializable]
    public class PlayerColors
    {
        public ColorSet Goalie;
        public ColorSet FieldPlayer;
    }

    [System.Serializable]
    public class ColorSet
    {
        public string Shirt;
        public string Shorts;
        public string Shoes;
    }

     public enum PlayerInfoDisplayMode
    {
        NumberOnly,
        NumberAndPosition,
        FullInfo
    }

public class PlayerEventInfo
{
    public int LeagueId;
    public string PlayerName;
    public string PlayerType;
    public int RowId;
    public string MatchClock;
}

    private Dictionary<string, Dictionary<string, List<EventPositionalData>>> eventData;
    private Dictionary<int, RootObject> eventKeyData = new Dictionary<int, RootObject>();
    private List<string> eventKeys;
    private List<string> timeKeys;
    private int currentEventKeyIndex = 0;
    private int currentTimeKeyIndex = 0;
    private int lastChooseEvent = 0;

    // Class definitons for the JSON data sets
    private Dictionary<int, List<ReducedData>> groupedData;
    //private Dictionary<int, List<EventPositionalData>> groupedEventData;
    
    private int currentTimeMsIndex = 0;
    private float elapsedTime = 0f;
    //private float updateInterval = 0.08f; // Update data once per second mit 0.5. 0.2 is faster
    private float gameTimer = 0f; // Game timer in seconds

    private float gameElapsedTime = 0f;
    private float gameStartTime = 0f;
    
    public enum ThresholdType
{
    V,
    LogV,
    SqrtV
}

public class PlayerVelocityText
{
    public TextMeshPro velocityText;
    public int playerIndex;
}

private List<PlayerVelocityText> playerVelocityTexts = new List<PlayerVelocityText>();

private ThresholdType currentThresholdType = ThresholdType.V;

    private Dictionary<int, Vector3> targetPositions;
    private Dictionary<int, Vector3> startPositions;
    private List<EventPositionalData> chosenEventData;
    
    private Dictionary<int, float> lerpTimes;
    private Dictionary<int, GameObject> arrows;
    private Dictionary<int, float> playerHeights; // Dictionary to map player IDs to their heights
    private Dictionary<int, float> playerWeights; // Dictionary to map player IDs to their weights
    private Dictionary<int, int> playerNumbers; // Dictionary to map player IDs to their numbers
    private Dictionary<int, GameObject> textObjects; // Dictionary to store the text objects
    private Dictionary<int, Vector3> lastPositions = new Dictionary<int, Vector3>();
    public TextMeshProUGUI timerText; // Reference to the TextMeshProUGUI component
    public TextMeshProUGUI gametimerText; // Reference to the TextMeshProUGUI component
    public TextMeshProUGUI currentEventKeyText; // Reference to the TextMeshProUGUI component for displaying current event key

    private GameObject[] balls;

    public JsonData clusterCenters;
    private bool showVelocityText = true;

    [SerializeField] private TMP_Dropdown teamDropdown;
    [SerializeField] private TMP_Dropdown playerDropdown;
    [SerializeField] private TMP_Dropdown eventDropdown;

    private Dictionary<string, List<PlayerInfo>> teamPlayers = new Dictionary<string, List<PlayerInfo>>();
    //private Dictionary<int, List<EventInfo>> playerEvents = new Dictionary<int, List<EventInfo>>();

    [SerializeField] private TMP_Dropdown gameModeDropdown;
    [SerializeField] private Button playButton;
    [SerializeField] private GameObject playerSelectionUI; // Parent object for team, player, event dropdowns

    private enum GameMode
    {
        PlayMatch,
        PlayAllEvents,
        ChoosePlayerEvents
    }

    public TMPro.TextMeshProUGUI thresholdModeText;
    public TMPro.TextMeshProUGUI thresholdValuesText;

    private GameMode selectedGameMode;

    private bool gameStarted = false;

    [SerializeField] private Button cancelButton;

    void Start()
{
    Debug.Log("Start method called");

    UpdateThresholdModeText();
    UpdateThresholdValuesText();
    
    Debug.Log("Created player velocity texts");
    
    try
    {
        InitializeDictionaries();
        LoadJsonData();
        if (myUniqueIdsList == null || myUniqueIdsList.league_teams == null || myUniqueIdsList.league_teams.Length == 0)
    {
        Debug.LogError("myUniqueIdsList is not properly loaded");
    }
    else
    {
        Debug.Log($"myUniqueIdsList loaded with {myUniqueIdsList.league_teams.Length} teams");
    }

    CreatePlayerVelocityTexts();
        StartCoroutine(LogPlayerPositions());
        LoadEventPositionalData();
        ProcessPlayerData();
        ProcessEventData();
        PopulateTeamPlayersData(); 
        CreateArrowsAndAdjustPlayers();
        CreateBorder();
        StartCoroutine(CheckInactivity());
        LoadAllEventKeyData();
        CreateIndexToLeagueIdMap();
        InitializeDropdowns();
        InitializeUI();
        LoadAllEventKeyData();

        LoadJerseyColors();
        if (jerseyColors != null && jerseyColors.Count > 0)
        {
            Debug.Log($"Loaded {jerseyColors.Count} teams' jersey colors. Applying colors...");
            ApplyJerseyColors();
        }
        else
        {
            Debug.LogWarning("jerseyColors is null or empty. Skipping ApplyJerseyColors.");
        }

        lastShowBallArrow = showBallArrow;
        lastShowPlayerArrows = showPlayerArrows;

        InitializeBalls();

        gameStartTime = Time.time;
        Debug.Log("Start method completed successfully");
    }
    catch (System.Exception e)
    {
        Debug.LogError($"Error in Start method: {e.Message}\n{e.StackTrace}");
    }
}

private void CreatePlayerVelocityTexts()
{Debug.Log($"Creating player velocity texts. League teams count: {myUniqueIdsList.league_teams.Length}");
    foreach (var uniqueId in myUniqueIdsList.league_teams)
    {Debug.Log($"Processing league_id: {uniqueId.league_id}, index: {uniqueId.index}");
        if (uniqueId.league_id > 3) // Assuming league_id 1, 2, 3 are balls
        {
            GameObject player = GameObject.Find("Player" + uniqueId.index);
            if (player != null)
            {
                GameObject textObj = new GameObject($"VelocityText_Player{uniqueId.index}");
                textObj.transform.SetParent(player.transform);
                textObj.transform.localPosition = new Vector3(0, 2, 0); // Adjust this to position the text above the player

                TextMeshPro tmpro = textObj.AddComponent<TextMeshPro>();
                tmpro.alignment = TextAlignmentOptions.Center;
                tmpro.fontSize = 4; // Increased font size
                tmpro.color = Color.white;
                tmpro.outlineWidth = 0.2f; // Add outline for better visibility
                tmpro.outlineColor = Color.black;

                // Ensure the text always faces the camera
                tmpro.gameObject.AddComponent<Billboard>();

                PlayerVelocityText pvt = new PlayerVelocityText
                {
                    velocityText = tmpro,
                    playerIndex = uniqueId.index
                };
                playerVelocityTexts.Add(pvt);
                
                // Set initial visibility
                pvt.velocityText.gameObject.SetActive(showVelocityText);

                Debug.Log($"Created velocity text for Player{uniqueId.index}");
            }
            else
            {
                Debug.LogWarning($"Player{uniqueId.index} not found");
            }
        }
    else
        {
            Debug.Log($"Skipping league_id {uniqueId.league_id} (assumed to be a ball)");
        }
    }
    Debug.Log($"Finished creating player velocity texts. Count: {playerVelocityTexts.Count}");
}

private void UpdatePlayerVelocityTexts()
{
    foreach (var pvt in playerVelocityTexts)
    {
        GameObject player = GameObject.Find("Player" + pvt.playerIndex);
        if (player != null && player.activeSelf)
        {
            float velocity = CalculatePlayerVelocity(pvt.playerIndex);
            
            // Get the current threshold type
            float selectedV = 0f;
            switch (currentThresholdType)
            {
                case ThresholdType.V:
                    selectedV = velocity;
                    break;
                case ThresholdType.LogV:
                    selectedV = Mathf.Log(velocity + 1);
                    break;
                case ThresholdType.SqrtV:
                    selectedV = Mathf.Sqrt(velocity);
                    break;
            }

            string unit = currentThresholdType == ThresholdType.V ? "m/s" : 
                          currentThresholdType == ThresholdType.LogV ? "log(m/s)" : "√(m/s)";
            pvt.velocityText.text = $"V: {selectedV:F2} {unit}";

            // Color the text based on the velocity
            ClusterCenters thresholds = GetSelectedThresholds();
            if (thresholds != null)
            {
                if (selectedV < thresholds.idle_threshold)
                    pvt.velocityText.color = Color.green;
                else if (selectedV < thresholds.walk_threshold)
                    pvt.velocityText.color = Color.yellow;
                else
                    pvt.velocityText.color = Color.red;
            }

            // Only update visibility if it's different from the toggle state
            if (pvt.velocityText.gameObject.activeSelf != showVelocityText)
            {
                pvt.velocityText.gameObject.SetActive(showVelocityText);
            }
        }
        else
        {
            pvt.velocityText.gameObject.SetActive(false);
        }
    }
}

private float CalculatePlayerVelocity(int playerIndex)
{
    if (targetPositions.TryGetValue(playerIndex, out Vector3 targetPos) &&
        startPositions.TryGetValue(playerIndex, out Vector3 startPos))
    {
        float velocity = Vector3.Distance(targetPos, startPos) / gameSpeed;
        Debug.Log($"Calculated velocity for Player{playerIndex}: {velocity}");
        return velocity;
    }
    Debug.Log($"Could not calculate velocity for Player{playerIndex}");
    return 0f;
}

void DebugPrintEventInfo()
{
    if (myEventInfoList == null || myEventInfoList.EventInfo == null)
    {
        Debug.LogError("myEventInfoList or its EventInfo is null");
        return;
    }

    Debug.Log($"Total events in myEventInfoList: {myEventInfoList.EventInfo.Count}");

    foreach (var eventInfo in myEventInfoList.EventInfo)
    {
        Debug.Log($"Event ID: {eventInfo.row_id}, Name: {eventInfo.event_name}, Time: {eventInfo.match_clock}");
        Debug.Log($"Players involved:");
        foreach (var player in eventInfo.players)
        {
            Debug.Log($"  Player ID: {player.player_id}, Name: {player.player_name}, Type: {player.player_type}");
        }
        Debug.Log("--------------------");
    }
}

private void SwitchThresholdType()
{
    currentThresholdType = (ThresholdType)(((int)currentThresholdType + 1) % 3);
    UpdateThresholdModeText();
    UpdateThresholdValuesText();
    Debug.Log($"Switched to {currentThresholdType} thresholds");
    UpdatePlayerAnimations(); 
}

private void UpdateThresholdModeText()
{
    if (thresholdModeText != null)
    {
        switch (currentThresholdType)
        {
            case ThresholdType.V:
                thresholdModeText.text = "Threshold Mode: V";
                break;
            case ThresholdType.LogV:
                thresholdModeText.text = "Threshold Mode: Log V";
                break;
            case ThresholdType.SqrtV:
                thresholdModeText.text = "Threshold Mode: Sqrt V";
                break;
        }
    }
    else
    {
        Debug.LogWarning("Threshold mode text is not assigned!");
    }
}

private void UpdateThresholdValuesText()
{
    if (thresholdValuesText != null)
    {
        ClusterCenters selectedThresholds = GetSelectedThresholds();
        if (selectedThresholds != null)
        {
            string text = string.Format("Thresholds:\nIdle: {0:F2}\nWalk: {1:F2}\nRun: {2:F2}",
                selectedThresholds.idle_threshold,
                selectedThresholds.walk_threshold,
                selectedThresholds.run_threshold);
            thresholdValuesText.text = text;
            Debug.Log($"Updated threshold values text: {text}");
        }
        else
        {
            thresholdValuesText.text = "Thresholds not available";
            Debug.LogError("Selected thresholds are null");
        }
    }
    else
    {
        Debug.LogWarning("Threshold values text is not assigned!");
    }
}

private List<string> DeterminePlayingTeams()
{
    HashSet<string> teams = new HashSet<string>();

    // Assuming myUniqueIdsList.league_teams contains the current match data
    foreach (var player in myUniqueIdsList.league_teams)
    {
        if (player.league_id > 3) // Skip balls (assuming league_id 1, 2, 3 are balls)
        {
            string teamName = GetTeamName(player.league_id);
            if (!string.IsNullOrEmpty(teamName))
            {
                teams.Add(teamName);
            }
        }
    }

    return new List<string>(teams);
}

private void OnDrawGizmos()
{
    if (Application.isPlaying)
    {
        foreach (var kvp in targetPositions)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawSphere(kvp.Value, 0.5f);
            Gizmos.color = Color.white;
            Gizmos.DrawLine(kvp.Value, kvp.Value + Vector3.up * 2);
#if UNITY_EDITOR
            UnityEditor.Handles.Label(kvp.Value + Vector3.up * 2, "Player" + kvp.Key);
#endif
        }
    }
}

public void DebugMovePlayer(int playerIndex, Vector3 newPosition)
{
    string playerName = "Player" + playerIndex;
    GameObject player = GameObject.Find(playerName);
    if (player != null)
    {
        player.transform.position = newPosition;
        Debug.Log($"Forcibly moved {playerName} to {newPosition}");
    }
    else
    {
        Debug.LogError($"Player {playerName} not found for debug move");
    }
}

void InitializeUI()
{
    if (gameModeDropdown == null || playButton == null || playerSelectionUI == null)
    {
        Debug.LogError("Some UI elements are not assigned in the Inspector!");
        return;
    }

    // Setup game mode dropdown
    //List<string> playingTeams = DeterminePlayingTeams();
    gameModeDropdown.ClearOptions();
    gameModeDropdown.AddOptions(new List<string> { "Play Match", "Play All Events", "Choose Player Events" });
    gameModeDropdown.onValueChanged.AddListener(OnGameModeChanged);

    // Setup play button
    playButton.onClick.RemoveAllListeners();
    playButton.onClick.AddListener(StartGame);

    // Setup cancel button
    if (cancelButton != null)
    {
        cancelButton.onClick.RemoveAllListeners();
        cancelButton.onClick.AddListener(CancelGame);
        cancelButton.gameObject.SetActive(false);
    }
    else
    {
        Debug.LogError("Cancel Button is not assigned in the Inspector!");
    }

    playerSelectionUI.SetActive(false);

    // Disable existing gameplay elements
    SetGameplayActive(false);

    // Set initial game mode and trigger OnGameModeChanged
    gameModeDropdown.value = 0; // This will select "Play Match"
    OnGameModeChanged(0);

    // Setup player dropdown
    playerDropdown.onValueChanged.AddListener(OnPlayerSelected);

    // Setup event dropdown
    eventDropdown.onValueChanged.AddListener(OnEventSelected);

    // Initialize team dropdown (but keep it hidden initially)
    List<string> playingTeams = DeterminePlayingTeams();
    teamDropdown.ClearOptions();
    teamDropdown.AddOptions(playingTeams);
    teamDropdown.onValueChanged.AddListener(OnTeamSelected);
    teamDropdown.gameObject.SetActive(false);

    // Initialize player and event dropdowns (keep them hidden initially)
    playerDropdown.gameObject.SetActive(false);
    eventDropdown.gameObject.SetActive(false);

    Debug.Log("UI Initialized successfully");
}

private IEnumerator PlayChosenEventCoroutine()
{
    Debug.Log($"Starting PlayChosenEventCoroutine for event {chooseEvent}");

    if (!eventKeyData.ContainsKey(chooseEvent))
    {
        Debug.LogError($"No data found for event {chooseEvent}");
        yield break;
    }

    while (gameStarted)
    {
        MovePlayersBasedOnChosenEvent();
        
        if (currentTimeKeyIndex >= timeKeys.Count)
        {
            Debug.Log($"Event {chooseEvent} playback completed");
            break;
        }

        yield return new WaitForSeconds(1f / eventSpeed);
    }

    Debug.Log("PlayChosenEventCoroutine finished");
    ResetChosenEvent();
}

void CancelGame()
{
    // Stop the game
    SetGameplayActive(false);
    gameStarted = false;

    // Reset game variables
    playMatch = 0;
    playAllEvents = 0;
    chooseEvent = 0;
    currentTimeMsIndex = 0;
    currentEventKeyIndex = 0;
    currentTimeKeyIndex = 0;

    // Stop all coroutines
    StopAllCoroutines();

    // Reset player positions
    ResetPlayerPositions();

    // Show game mode selection UI
    gameModeDropdown.gameObject.SetActive(true);
    playButton.gameObject.SetActive(true);
    cancelButton.gameObject.SetActive(false);

    // Reset dropdowns
    ResetDropdowns();

    Debug.Log("Game cancelled and returned to mode selection");
}

public void ToggleVelocityTextVisibility()
{
    showVelocityText = !showVelocityText;
    foreach (var pvt in playerVelocityTexts)
    {
        if (pvt.velocityText != null)
        {
            pvt.velocityText.gameObject.SetActive(showVelocityText);
        }
    }
    Debug.Log($"Velocity text visibility: {(showVelocityText ? "On" : "Off")}");
}

void ResetPlayerPositions()
{
    foreach (var kvp in targetPositions)
    {
        int index = kvp.Key;
        string playerName = "Player" + index;
        GameObject player = GameObject.Find(playerName);
        if (player != null)
        {
            player.transform.position = Vector3.zero;
        }
    }
}

    void ResetDropdowns()
    {
        gameModeDropdown.value = 0;
        playerSelectionUI.SetActive(false);
        playButton.interactable = false;
    }

void OnGameModeChanged(int index)
{
    selectedGameMode = (GameMode)index;
    playButton.interactable = true;

    bool isChoosePlayerMode = selectedGameMode == GameMode.ChoosePlayerEvents;
    playerSelectionUI.SetActive(isChoosePlayerMode);

    if (isChoosePlayerMode)
    {
        PopulateTeamDropdown();
        teamDropdown.gameObject.SetActive(true);
        playerDropdown.gameObject.SetActive(false);
        eventDropdown.gameObject.SetActive(false);
    }
    else
    {
        teamDropdown.gameObject.SetActive(false);
        playerDropdown.gameObject.SetActive(false);
        eventDropdown.gameObject.SetActive(false);
    }

    Debug.Log($"Game mode changed to: {selectedGameMode}. Play button interactable: {playButton.interactable}");
}

void StartGame()
{
    Debug.Log($"StartGame called. Selected game mode: {selectedGameMode}");

    // Hide UI elements
    gameModeDropdown.gameObject.SetActive(false);
    playButton.gameObject.SetActive(false);
    playerSelectionUI.SetActive(false);

    // Show cancel button
    cancelButton.gameObject.SetActive(true);

    // Enable gameplay elements
    SetGameplayActive(true);

    // Start the appropriate game mode
    switch (selectedGameMode)
    {
        case GameMode.PlayMatch:
            StartPlayMatch();
            break;
        case GameMode.PlayAllEvents:
            StartPlayAllEvents();
            break;
        case GameMode.ChoosePlayerEvents:
            if (!eventSelected)
            {
                Debug.LogWarning("No event selected for Choose Player Events mode.");
                return;
            }
            StartChoosePlayerEvents();
            break;
    }

    gameStarted = true;
    Debug.Log("Game started successfully");
}

    void SetGameplayActive(bool active)
{
    // Enable/disable your existing gameplay elements here
    playMatch = active ? 1 : 0;
    playAllEvents = active ? 1 : 0;
    
    // Enable/disable player movement
    foreach (var kvp in targetPositions)
    {
        int index = kvp.Key;
        string playerName = "Player" + index;
        GameObject player = GameObject.Find(playerName);
        if (player != null)
        {
            Rigidbody rb = player.GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.isKinematic = !active; // Set to kinematic when not active
            }
        }
    }

    // Reset game timer if deactivating
    if (!active)
    {
        gameTimer = 0f;
        if (gametimerText != null)
            gametimerText.text = "Time: 0s";
    }

    Debug.Log($"Gameplay active: {active}");
}

void StartPlayMatch()
{
    Debug.Log("StartPlayMatch called");
    playMatch = 1;
    playAllEvents = 0;
    chooseEvent = 0;
    currentTimeMsIndex = 0; // Reset to start of the match
    // Additional initialization for match play
    Debug.Log("Match play mode initialized");
}

void StartPlayAllEvents()
{
    playMatch = 0;
    playAllEvents = 1;
    chooseEvent = 0;
    currentEventKeyIndex = 0; // Reset to first event
    currentTimeKeyIndex = 0;
    // Additional initialization for all events play
}

void InitializeDropdowns()
{
    teamDropdown.onValueChanged.AddListener(OnTeamSelected);
    playerDropdown.onValueChanged.AddListener(OnPlayerSelected);
    eventDropdown.onValueChanged.AddListener(OnEventSelected);

    // Initially hide all dropdowns
    teamDropdown.gameObject.SetActive(false);
    playerDropdown.gameObject.SetActive(false);
    eventDropdown.gameObject.SetActive(false);
}

void PopulateEventDropdown(string team, string player)
{
    eventDropdown.ClearOptions();
    List<string> eventDescriptions = GetEventsForPlayer(team, player);
    Debug.Log($"Populating event dropdown with {eventDescriptions.Count} events for {player} from {team}");
    eventDropdown.AddOptions(eventDescriptions);
    
    if (eventDescriptions.Count > 0)
    {
        eventDropdown.value = 0;
        eventDropdown.RefreshShownValue();
    }
    else
    {
        Debug.LogWarning($"No events found for player {player} from team {team}");
    }
}

List<string> GetEventsForPlayer(string team, string player)
{
    List<string> eventDescriptions = new List<string>();
    
    Debug.Log($"Searching events for player: {player} from team: {team}");

    // Find the player's league ID
    int leagueId = -1;
    foreach (var playerInfo in myPlayerInfoList.PlayerInfo)
    {
        if (playerInfo.Team == team && $"{playerInfo.First_Name} {playerInfo.Last_Name}" == player)
        {
            leagueId = playerInfo.ID;
            break;
        }
    }

    if (leagueId != -1 && playerEvents.ContainsKey(leagueId))
    {
        foreach (var eventInfo in playerEvents[leagueId])
        {
            string description = $"[{eventInfo.RowId}] {eventInfo.PlayerType} at {eventInfo.MatchClock}";
            eventDescriptions.Add(description);
            Debug.Log($"Added event: {description}");
        }
    }
    else
    {
        Debug.LogWarning($"No events found for player: {player} (League ID: {leagueId})");
    }

    Debug.Log($"Total events found for player: {eventDescriptions.Count}");
    return eventDescriptions;
}

void OnTeamSelected(int index)
{
    string selectedTeam = teamDropdown.options[index].text;
    Debug.Log($"Selected team: '{selectedTeam}'");
    PopulatePlayerDropdown(selectedTeam);
    playerDropdown.interactable = true;
    playerDropdown.gameObject.SetActive(true);
    eventDropdown.interactable = false;
    eventDropdown.gameObject.SetActive(false);
    eventDropdown.ClearOptions();
}

void OnPlayerSelected(int index)
{
    string selectedTeam = teamDropdown.options[teamDropdown.value].text;
    string selectedPlayer = playerDropdown.options[index].text;
    PopulateEventDropdown(selectedTeam, selectedPlayer);
    eventDropdown.interactable = true;
    eventDropdown.gameObject.SetActive(true);
}

void OnEventSelected(int index)
{
    string selectedEvent = eventDropdown.options[index].text;
    int eventId = ExtractEventIdFromString(selectedEvent);
    SetupChosenEvent(eventId);
}

void SetupChosenEvent(int eventId)
{
    chooseEvent = eventId;
    eventSelected = true;
    Debug.Log($"Event selected: {eventId}");
    // You might want to enable the play button or take other actions here
}

int ExtractEventIdFromString(string eventDescription)
{
    // Assuming the event ID is enclosed in square brackets at the start of the description
    int startIndex = eventDescription.IndexOf('[') + 1;
    int endIndex = eventDescription.IndexOf(']');
    if (startIndex > 0 && endIndex > startIndex)
    {
        string eventIdStr = eventDescription.Substring(startIndex, endIndex - startIndex);
        if (int.TryParse(eventIdStr, out int eventId))
        {
            return eventId;
        }
    }
    Debug.LogWarning($"Failed to extract event ID from: {eventDescription}");
    return -1;
}

void PopulateTeamDropdown()
{
    List<string> playingTeams = DeterminePlayingTeams();

    Debug.Log($"Found {playingTeams.Count} teams playing:");
    foreach (var team in playingTeams)
    {
        Debug.Log($"Team: '{team}'");
    }

    teamDropdown.ClearOptions();
    teamDropdown.AddOptions(playingTeams);
    teamDropdown.value = 0;
    teamDropdown.RefreshShownValue();
}

void ResetChosenEvent()
{
    currentTimeKeyIndex = 0;
    // Don't set timeKeys to null here
    // timeKeys = null;
    StopCoroutine(PlayChosenEvent());
    gameStarted = false;
    Debug.Log("ResetChosenEvent called");
}

void PopulateTeamPlayersData()
{
    teamPlayers.Clear();
    foreach (var player in myPlayerInfoList.PlayerInfo)
    {
        string normalizedTeamName = NormalizeTeamName(player.Team);
        if (!teamPlayers.ContainsKey(normalizedTeamName))
        {
            teamPlayers[normalizedTeamName] = new List<PlayerInfo>();
        }
        teamPlayers[normalizedTeamName].Add(player);
        Debug.Log($"Added player {player.First_Name} {player.Last_Name} (ID: {player.ID}) to team '{normalizedTeamName}' (original: '{player.Team}')");
    }
    Debug.Log($"Populated team players data. Total teams: {teamPlayers.Count}");
    foreach (var team in teamPlayers.Keys)
    {
        Debug.Log($"Team: '{team}', Players: {teamPlayers[team].Count}");
    }
}

void PopulatePlayerDropdown(string team)
{
    playerDropdown.ClearOptions();
    Debug.Log($"Populating player dropdown for team: '{team}'");
    
    List<string> playerNames = new List<string>();
    foreach (var playerInfo in myPlayerInfoList.PlayerInfo)
    {
        if (playerInfo.Team == team)
        {
            string playerName = $"{playerInfo.First_Name} {playerInfo.Last_Name}";
            playerNames.Add(playerName);
        }
    }
    playerNames.Sort();
    
    playerDropdown.AddOptions(playerNames);
    playerDropdown.value = 0;
    playerDropdown.RefreshShownValue();
    
    Debug.Log($"Added {playerNames.Count} players to dropdown for team {team}");
}

string NormalizeTeamName(string teamName)
{
    return teamName.Trim().ToLowerInvariant().Replace(" ", "");
}

private bool eventSelected = false;

void OnEventSelected(TMP_Dropdown dropdown)
{
    int eventIndex = dropdown.value;
    List<EventInfo> playerEvents = dropdown.gameObject.GetData<List<EventInfo>>("playerEvents");

    if (playerEvents != null && eventIndex < playerEvents.Count)
    {
        EventInfo selectedEvent = playerEvents[eventIndex];
        chooseEvent = selectedEvent.row_id;
        eventSelected = true;
        Debug.Log($"Selected event: {selectedEvent.event_name} (row_id: {chooseEvent})");
    }
    else
    {
        Debug.LogError("Failed to retrieve selected event information");
        eventSelected = false;
    }
}

EventInfo FindSelectedEvent(int playerId, int eventIndex)
{
    int currentIndex = 0;
    foreach (var eventInfo in myEventInfoList.EventInfo)
    {
        foreach (var player in eventInfo.players)
        {
            if (player.player_id == playerId)
            {
                if (currentIndex == eventIndex)
                {
                    return eventInfo;
                }
                currentIndex++;
                break;
            }
        }
    }
    return null;
}
void StartChoosePlayerEvents()
{
    if (!eventSelected)
    {
        Debug.LogWarning("No event selected. Please select an event before starting.");
        return;
    }

    playMatch = 0;
    playAllEvents = 0;
    currentEventKeyIndex = 0;
    currentTimeKeyIndex = 0;
    timeKeys = null;

    // Hide UI elements
    gameModeDropdown.gameObject.SetActive(false);
    playButton.gameObject.SetActive(false);

    // Show cancel button
    cancelButton.gameObject.SetActive(true);

    // Enable gameplay elements
    SetGameplayActive(true);

    gameStarted = true;
    Debug.Log($"Chosen player event started. Event row_id: {chooseEvent}");

    // Start the event playback
    StartCoroutine(PlayChosenEventCoroutine());
}

private IEnumerator LogPlayerPositions()
{
    while (true)
    {
        yield return new WaitForSeconds(1f); // Log every second

        foreach (var kvp in targetPositions)
        {
            int index = kvp.Key;
            string playerName = "Player" + index;
            GameObject player = GameObject.Find(playerName);
            if (player != null)
            {
                Debug.Log($"{playerName} position: {player.transform.position}, Target: {kvp.Value}");
            }
        }
    }
}



IEnumerator PlayChosenEvent()
{
    Debug.Log($"Starting PlayChosenEvent for event {chooseEvent}");

    if (!eventKeyData.ContainsKey(chooseEvent))
    {
        Debug.LogError($"No data found for event {chooseEvent}");
        yield break;
    }

    // Initialize timeKeys if it's null
    if (timeKeys == null)
    {
        var eventDataToUse = eventKeyData[chooseEvent].event_data;
        if (eventDataToUse == null || eventDataToUse.Count == 0)
        {
            Debug.LogError($"Event data is empty for event {chooseEvent}");
            yield break;
        }

        string currentEventKey = eventDataToUse.Keys.First();
        var timeData = eventDataToUse[currentEventKey];
        
        timeKeys = new List<string>(timeData.Keys);
        timeKeys.Sort((a, b) => int.Parse(a).CompareTo(int.Parse(b)));
        currentTimeKeyIndex = 0;
        
        Debug.Log($"Initialized timeKeys for event {chooseEvent}. Count: {timeKeys.Count}");
    }

    while (true)
    {
        MovePlayersBasedOnChosenEvent();
        yield return new WaitForSeconds(gameSpeed);

        // Check if the event has completed
        if (timeKeys != null && currentTimeKeyIndex >= timeKeys.Count)
        {
            Debug.Log($"Event {chooseEvent} playback completed");
            break;
        }
        else if (timeKeys == null)
        {
            Debug.LogError($"timeKeys is null for event {chooseEvent}");
            break;
        }
    }

    // Reset after playback is complete
    ResetChosenEvent();
}

    // This method should be called after loading your event data
private Dictionary<int, List<PlayerEventInfo>> playerEvents = new Dictionary<int, List<PlayerEventInfo>>();


void ProcessEventData()
{
    playerEvents.Clear();
    Debug.Log($"Processing {myEventInfoList.EventInfo.Count} events");

    foreach (var eventInfo in myEventInfoList.EventInfo)
    {
        if (eventInfo.players == null || eventInfo.players.Count == 0)
        {
            Debug.Log($"Skipping event {eventInfo.row_id} as it has no players");
            continue;
        }

        foreach (var player in eventInfo.players)
        {
            if (player.player_id <= 0)
            {
                Debug.LogWarning($"Invalid player_id for player {player.player_name} in event {eventInfo.row_id}");
                continue;
            }

            PlayerEventInfo playerEventInfo = new PlayerEventInfo
            {
                LeagueId = player.player_id,
                PlayerName = player.player_name,
                PlayerType = player.player_type,
                RowId = eventInfo.row_id,
                MatchClock = eventInfo.match_clock
            };

            if (!playerEvents.ContainsKey(player.player_id))
            {
                playerEvents[player.player_id] = new List<PlayerEventInfo>();
            }
            playerEvents[player.player_id].Add(playerEventInfo);

            Debug.Log($"Added event for player {player.player_name} (ID: {player.player_id}): Event {eventInfo.row_id}, Type: {player.player_type}");
        }
    }

    Debug.Log($"Processed events for {playerEvents.Count} unique players");
    foreach (var kvp in playerEvents)
    {
        Debug.Log($"Player ID {kvp.Key}: {kvp.Value.Count} events");
    }
}
    int MapEventToPlayerId(EventInfo eventInfo)
    {
        // Implement this method based on how your event data is structured
        // It should return the player ID associated with the event
        // This is a placeholder implementation
        return 0;
    }


private void InitializeBalls()
{
    balls = new GameObject[3];
    for (int i = 0; i < 3; i++)
    {
        balls[i] = GameObject.Find($"Player{300 + i}");
        if (balls[i] == null)
        {
            Debug.LogWarning($"Ball object Player{300 + i} not found");
        }
    }

    if (balls.All(b => b == null))
    {
        Debug.LogError("No ball objects found in the scene. Make sure they're named 'Player300', etc.");
    }
}

    void InitializeDictionaries()
    { // Initialize the dictionaries
        targetPositions = new Dictionary<int, Vector3>();
        startPositions = new Dictionary<int, Vector3>();
        lerpTimes = new Dictionary<int, float>();
        arrows = new Dictionary<int, GameObject>();
        playerHeights = new Dictionary<int, float>();
        playerWeights = new Dictionary<int, float>();
        playerNumbers = new Dictionary<int, int>();
        textObjects = new Dictionary<int, GameObject>();
        lastMovedTimes = new Dictionary<int, float>();
        teamIdToName = new Dictionary<int, string>();
        playerPositions = new Dictionary<int, string>();
        leagueIdToTeamMap = new Dictionary<int, string>();
    }

string GetTeamName(int leagueId)
{
    if (leagueIdToTeamMap.TryGetValue(leagueId, out string teamName))
    {
        return teamName;
    }
    Debug.LogWarning($"Team name not found for league ID: {leagueId}");
    return string.Empty;
}

   void LoadJerseyColors()
    {
        Debug.Log("LoadJerseyColors method called");
        if (jerseyColorsJson == null || jsonFile_player_info == null)
        {
            Debug.LogError("jerseyColorsJson or jsonFile_player_info is null. Please assign them in the Unity Inspector.");
            return;
        }

        // Load jersey colors
        string jerseyJsonContent = jerseyColorsJson.text;
        Debug.Log($"Jersey colors JSON content: {jerseyJsonContent}");

        try
        {
            jerseyColors = JsonConvert.DeserializeObject<JerseyColors>(jerseyJsonContent);
            
            if (jerseyColors == null || jerseyColors.Count == 0)
            {
                Debug.LogError("Failed to parse jersey color data or the dictionary is empty.");
                return;
            }

            Debug.Log($"Successfully loaded jersey colors for {jerseyColors.Count} teams.");
            foreach (var team in jerseyColors.Keys)
            {
                Debug.Log($"Loaded colors for team: {team}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error parsing jersey color JSON: {e.Message}");
            return;
        }

        // Load player info
        string playerInfoJsonContent = jsonFile_player_info.text;
        PlayerInfoList playerInfoList = JsonUtility.FromJson<PlayerInfoList>(playerInfoJsonContent);

        if (playerInfoList == null || playerInfoList.PlayerInfo == null || playerInfoList.PlayerInfo.Length == 0)
        {
            Debug.LogError("Failed to parse player info data or the list is empty.");
            return;
        }

        // Create mapping between team IDs and team names
        foreach (var player in playerInfoList.PlayerInfo)
        {
            if (!teamIdToName.ContainsKey(player.ID))
            {
                teamIdToName[player.ID] = player.Team;
            }
        }

        Debug.Log($"Mapped {teamIdToName.Count} team IDs to team names.");
    }

    void ApplyJerseyColors()
{
    if (jerseyColors == null || jerseyColors.Count == 0 || teamIdToName == null || teamIdToName.Count == 0)
    {
        Debug.LogError("Jersey colors data or team mapping not loaded. Cannot apply jersey colors.");
        return;
    }

    Debug.Log($"Applying jersey colors. Player count: {myUniqueIdsList.league_teams.Length}, PlayerPositions count: {playerPositions.Count}");

    foreach (var player in myUniqueIdsList.league_teams)
    {
        // Skip balls (league IDs 1, 2, and 3)
        if (player.league_id <= 3)
        {
            Debug.Log($"Skipping ball with league ID {player.league_id}");
            continue;
        }

        Debug.Log($"Processing player: League ID {player.league_id}, Team {player.team}, Index {player.index}");

        if (!teamIdToName.TryGetValue(player.league_id, out string teamName))
        {
            Debug.LogWarning($"No team name found for player with league ID {player.league_id}. Skipping.");
            continue;
        }

        if (!jerseyColors.TryGetValue(teamName, out TeamColors teamColors))
        {
            Debug.LogWarning($"No jersey colors found for team {teamName}. Skipping.");
            continue;
        }

        bool isGoalie = false;
        if (playerPositions.TryGetValue(player.league_id, out string position))
        {
            isGoalie = position == "G";
            Debug.Log($"Player {player.league_id} position: {position}, isGoalie: {isGoalie}");
        }
        else
        {
            Debug.LogWarning($"No position found for player with league ID {player.league_id}. Assuming not a goalie.");
        }

        bool isHomeTeam = player.team == 1; // Assuming team 1 is home team

        ColorSet colorSet;
        if (isHomeTeam)
        {
            colorSet = isGoalie ? teamColors.Home.Goalie : teamColors.Home.FieldPlayer;
        }
        else
        {
            colorSet = isGoalie ? teamColors.Away.Goalie : teamColors.Away.FieldPlayer;
        }

        Debug.Log($"Applying colors to player {player.league_id}: Shirt {colorSet.Shirt}, Shorts {colorSet.Shorts}");
        ApplyColorToPlayer(player.index, colorSet);
    }

    Debug.Log("Jersey colors applied successfully.");
}
void ApplyColorToPlayer(int playerIndex, ColorSet colorSet)
{
    GameObject player = GameObject.Find("Player" + playerIndex);
    if (player != null)
    {
        Renderer[] renderers = player.GetComponentsInChildren<Renderer>();
        foreach (Renderer renderer in renderers)
        {
            if (renderer.name.Contains("Shirt"))
            {
                ApplyColorToRenderer(renderer, colorSet.Shirt);
                Debug.Log($"Applied shirt color {colorSet.Shirt} to Player {playerIndex}");
            }
            else if (renderer.name.Contains("Shorts"))
            {
                ApplyColorToRenderer(renderer, colorSet.Shorts);
                Debug.Log($"Applied shorts color {colorSet.Shorts} to Player {playerIndex}");
            }
            else if (renderer.name.Contains("Sneakers"))
            {
                ApplyColorToRenderer(renderer, colorSet.Shoes);
                Debug.Log($"Applied shoes color {colorSet.Shoes} to Player {playerIndex}");
            }
        }
    }
    else
    {
        Debug.LogWarning($"Player object not found: Player {playerIndex}");
    }
}

void ApplyColorToRenderer(Renderer renderer, string colorString)
{
    Color color;
    if (ColorUtility.TryParseHtmlString(colorString, out color))
    {
        foreach (Material material in renderer.materials)
        {
            material.color = color;
        }
    }
    else
    {
        Debug.LogWarning($"Failed to parse color: {colorString}");
    }
}



    void LoadJsonData()
{
    Debug.Log("Starting to load JSON data...");

    try
    {
        LoadEventInfo();
        LoadArrayReduced();
        LoadIndexArray();
        LoadPlayerInfo();
        LoadEventPositionalInfo();
        CreateGroupedData();
        LoadEventPositionalData();

        Debug.Log("Finished loading JSON data.");
    }
    catch (System.Exception e)
    {
        Debug.LogError($"Error in LoadJsonData: {e.Message}\n{e.StackTrace}");
    }
}

void LoadEventInfo()
{
    if (jsonFile_event_info == null)
    {
        Debug.LogError("jsonFile_event_info is null");
        return;
    }

    try
    {
        string jsonContent = jsonFile_event_info.text;
        Debug.Log($"Event info JSON content (first 1000 characters): {jsonContent.Substring(0, Mathf.Min(1000, jsonContent.Length))}");
        
        myEventInfoList = JsonConvert.DeserializeObject<EventInfoList>(jsonContent);
        
        if (myEventInfoList != null && myEventInfoList.EventInfo != null)
        {
            Debug.Log($"Loaded jsonFile_event_info. Event info count: {myEventInfoList.EventInfo.Count}");
            Debug.Log($"First event: {myEventInfoList.EventInfo[0].event_name}, Players: {myEventInfoList.EventInfo[0].players.Count}");
        }
        else
        {
            Debug.LogError("Failed to deserialize event info or the list is empty.");
        }
    }
    catch (System.Exception e)
    {
        Debug.LogError($"Error loading event info: {e.Message}");
    }
}

void LoadArrayReduced()
{
    if (jsonFile_array_reduced == null)
    {
        Debug.LogError("jsonFile_array_reduced is null");
        return;
    }

    try
    {
        string jsonContent = jsonFile_array_reduced.text;
        JsonData jsonData = JsonUtility.FromJson<JsonData>(jsonContent);
        
        if (jsonData == null)
        {
            Debug.LogError("Failed to deserialize jsonFile_array_reduced");
            return;
        }

        myDataList = new DataList { data = jsonData.data };
        clusterCenters = jsonData;

        Debug.Log($"Loaded jsonFile_array_reduced. Data count: {myDataList?.data?.Length ?? 0}");
        
        LogClusterCenters();
    }
    catch (System.Exception e)
    {
        Debug.LogError($"Error loading array reduced: {e.Message}");
    }
}

void LogClusterCenters()
{
    Debug.Log("Logging Cluster Centers:");
    if (clusterCenters.cluster_centers_v != null)
        Debug.Log($"V: Idle={clusterCenters.cluster_centers_v.idle_threshold}, Walk={clusterCenters.cluster_centers_v.walk_threshold}, Run={clusterCenters.cluster_centers_v.run_threshold}");
    else
        Debug.LogError("cluster_centers_v is null");

    if (clusterCenters.cluster_centers_log_v != null)
        Debug.Log($"Log V: Idle={clusterCenters.cluster_centers_log_v.idle_threshold}, Walk={clusterCenters.cluster_centers_log_v.walk_threshold}, Run={clusterCenters.cluster_centers_log_v.run_threshold}");
    else
        Debug.LogError("cluster_centers_log_v is null");

    if (clusterCenters.cluster_centers_sqrt_v != null)
        Debug.Log($"Sqrt V: Idle={clusterCenters.cluster_centers_sqrt_v.idle_threshold}, Walk={clusterCenters.cluster_centers_sqrt_v.walk_threshold}, Run={clusterCenters.cluster_centers_sqrt_v.run_threshold}");
    else
        Debug.LogError("cluster_centers_sqrt_v is null");
}


void LoadIndexArray()
{
    if (jsonFile_index_array == null)
    {
        Debug.LogError("jsonFile_index_array is null");
        return;
    }

    try
    {
        string jsonContent = jsonFile_index_array.text;
        myUniqueIdsList = JsonUtility.FromJson<UniqueIdsList>(jsonContent);
        
        if (myUniqueIdsList != null && myUniqueIdsList.league_teams != null)
        {
            Debug.Log($"Loaded jsonFile_index_array. League teams count: {myUniqueIdsList.league_teams.Length}");
        }
        else
        {
            Debug.LogError("Failed to deserialize index array or the list is empty.");
        }
    }
    catch (System.Exception e)
    {
        Debug.LogError($"Error loading index array: {e.Message}");
    }
}

void LoadPlayerInfo()
{
    if (jsonFile_player_info == null)
    {
        Debug.LogError("jsonFile_player_info is null");
        return;
    }

    try
    {
        string jsonContent = jsonFile_player_info.text;
        myPlayerInfoList = JsonUtility.FromJson<PlayerInfoList>(jsonContent);
        
        if (myPlayerInfoList != null && myPlayerInfoList.PlayerInfo != null)
        {
            Debug.Log($"Loaded jsonFile_player_info. Player info count: {myPlayerInfoList.PlayerInfo.Length}");
        }
        else
        {
            Debug.LogError("Failed to deserialize player info or the list is empty.");
        }
    }
    catch (System.Exception e)
    {
        Debug.LogError($"Error loading player info: {e.Message}");
    }
}

void LoadEventPositionalInfo()
{
    if (jsonFile_event_positional_info == null)
    {
        Debug.LogError("jsonFile_event_positional_info is null");
        return;
    }

    try
    {
        string jsonContent = jsonFile_event_positional_info.text;
        myEventPositionalDataList = JsonUtility.FromJson<EventPositionalDataList>(jsonContent);
        
        if (myEventPositionalDataList != null && myEventPositionalDataList.event_data != null)
        {
            Debug.Log($"Loaded jsonFile_event_positional_info. Event data count: {myEventPositionalDataList.event_data.Length}");
        }
        else
        {
            Debug.LogError("Failed to deserialize event positional info or the list is empty.");
        }
    }
    catch (System.Exception e)
    {
        Debug.LogError($"Error loading event positional info: {e.Message}");
    }
}




void CreateGroupedData()
{
    if (myDataList != null && myDataList.data != null)
    {
        groupedData = myDataList.data.GroupBy(d => d.time_ms).ToDictionary(g => g.Key, g => g.ToList());
        Debug.Log($"Grouped data created. Group count: {groupedData.Count}");
    }
    else
    {
        Debug.LogWarning("Unable to create grouped data: myDataList or myDataList.data is null");
    }
}

    void LoadEventPositionalData()
    {
        string jsonFilePath = Path.Combine(Application.dataPath, "Ressources", "grouped_data_by_match_time.json");
        if (File.Exists(jsonFilePath))
        {
            string jsonContent = File.ReadAllText(jsonFilePath);
            RootObject rootObject = JsonConvert.DeserializeObject<RootObject>(jsonContent);
            if (rootObject != null && rootObject.event_data != null)
            {
                eventData = rootObject.event_data;
                eventKeys = new List<string>(eventData.Keys);
                Debug.Log("Loaded event data. Event count: " + eventKeys.Count);
                eventKeys.Sort((a, b) => int.Parse(a).CompareTo(int.Parse(b)));
            }
            else
            {
                Debug.LogError("Failed to deserialize JSON or event_data is null");
            }
        }
        else
        {
            Debug.LogError("JSON file not found at path: " + jsonFilePath);
        }
    }

void LoadAllEventKeyData()
{
    string folderPath = Path.Combine(Application.dataPath, "Ressources", "EventKeyData");
    if (Directory.Exists(folderPath))
    {
        string[] files = Directory.GetFiles(folderPath, "event_*.json");
        foreach (string filePath in files)
        {
            string fileName = Path.GetFileNameWithoutExtension(filePath);
            if (int.TryParse(fileName.Substring(6), out int eventNumber))
            {
                string jsonContent = File.ReadAllText(filePath);
                RootObject rootObject = JsonConvert.DeserializeObject<RootObject>(jsonContent);
                if (rootObject != null && rootObject.event_data != null)
                {
                    eventKeyData[eventNumber] = rootObject;
                    Debug.Log($"Loaded {fileName}.json with event number: {eventNumber}, Data count: {rootObject.event_data.Count}");
                }
                else
                {
                    Debug.LogWarning($"Failed to deserialize {fileName}.json or event_data is null");
                }
            }
            else
            {
                Debug.LogWarning($"Invalid file name format: {fileName}.json");
            }
        }
        Debug.Log($"Loaded {eventKeyData.Count} event files");
    }
    else
    {
        Debug.LogError("EventKeyData folder not found");
    }
}

    void ProcessPlayerData()
    {
        if (myPlayerInfoList != null && myPlayerInfoList.PlayerInfo != null)
        {
            foreach (var player in myPlayerInfoList.PlayerInfo)
            {
                playerHeights[player.ID] = player.Height;
                playerWeights[player.ID] = player.Weight;
                playerNumbers[player.ID] = player.Number;
                leagueIdToTeamMap[player.ID] = player.Team;
                playerPositions[player.ID] = player.Position;
            }
            Debug.Log($"Player data processed. Total players: {myPlayerInfoList.PlayerInfo.Length}");
            Debug.Log($"playerPositions count: {playerPositions.Count}");
            Debug.Log($"leagueIdToTeamMap count: {leagueIdToTeamMap.Count}");
        }
        else
        {
            Debug.LogWarning("Player info is null or empty.");
        }
    }

// You may want to add a method to get a player's position
public string GetPlayerPosition(int leagueId)
{
    if (playerPositions.TryGetValue(leagueId, out string position))
    {
        return position;
    }
    Debug.LogWarning($"Position not found for league ID: {leagueId}");
    return string.Empty;
}


private void SetPlayerVisibility(int index, bool isVisible)
{
    GameObject player = GameObject.Find("Player" + index);
    if (player != null)
    {
        Renderer renderer = player.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.enabled = isVisible;
        }

        // Hide/show arrow
        if (arrows.ContainsKey(index))
        {
            arrows[index].SetActive(isVisible);
        }

        // Hide/show text
        if (textObjects.ContainsKey(index))
        {
            textObjects[index].SetActive(isVisible);
        }
    }
}

    void RotatePlayerTowardsBall(int playerId)
{
    GameObject player = GameObject.Find("Player" + playerId);
    if (player == null) return;

    GameObject targetBall = null;

    if (selectedBallIndex == 0)
    {
        // Find the nearest ball
        float closestDistance = float.MaxValue;
        foreach (GameObject ball in balls)
        {
            if (ball != null)
            {
                float distance = Vector3.Distance(player.transform.position, ball.transform.position);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    targetBall = ball;
                }
            }
        }
    }
    else if (selectedBallIndex >= 1 && selectedBallIndex <= 3)
    {
        // Select the specific ball
        targetBall = balls[selectedBallIndex - 1];
    }

    if (targetBall != null)
    {
        Vector3 directionToBall = targetBall.transform.position - player.transform.position;
        directionToBall.y = 0; // Ignore vertical difference

        if (directionToBall != Vector3.zero)
        {
            Quaternion targetRotation = Quaternion.LookRotation(directionToBall);
            player.transform.rotation = targetRotation;
        }
    }
}


    void CreateArrowsAndAdjustPlayers()
{
    Debug.Log("CreateArrowsAndAdjustPlayers called.");
    if (myUniqueIdsList != null && myUniqueIdsList.league_teams != null)
    {
        foreach (var item in myUniqueIdsList.league_teams)
        {
            targetPositions[item.index] = Vector3.zero;
            startPositions[item.index] = Vector3.zero;
            lerpTimes[item.index] = 0f;

            CreateArrow(item.index);
            AdjustPlayer(item.index, item.league_id);
            
            // Set initial visibility
            SetInitialPlayerVisibility(item.index);
        }
    }

    // Create arrows for ball objects
    for (int i = 300; i <= 302; i++)
    {
        CreateArrow(i);
        SetInitialPlayerVisibility(i);
    }

    Debug.Log("Arrows and players created.");
}

private void SetInitialPlayerVisibility(int index)
{
    GameObject player = GameObject.Find("Player" + index);
    if (player != null)
    {
        Renderer renderer = player.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.enabled = false;
        }
        if (arrows.ContainsKey(index))
        {
            arrows[index].SetActive(false);
        }
        if (textObjects.ContainsKey(index))
        {
            textObjects[index].SetActive(false);
        }
    }
}

    private void UpdatePlayerArrowVisibility()
{
    foreach (var kvp in arrows)
    {
        int index = kvp.Key;
        GameObject arrow = kvp.Value;
        
        if (index < 300 || index > 302) // Not a ball
        {
            arrow.SetActive(showPlayerArrows);
        }
    }
}

    private void UpdateBallArrowVisibility()
{
    for (int i = 300; i <= 302; i++)
    {
        if (arrows.ContainsKey(i) && arrows[i] != null)
        {
            arrows[i].SetActive(showBallArrow);
        }
        else if (showBallArrow)
        {
            // Create arrow for ball if it doesn't exist and showBallArrow is true
            GameObject ball = GameObject.Find("Player" + i);
            if (ball != null)
            {
                GameObject arrow = Instantiate(arrowPrefab);
                arrow.transform.SetParent(ball.transform, false);
                arrow.transform.localPosition = new Vector3(0, -0.25f, 0);
                arrow.transform.localRotation = Quaternion.Euler(90, 0, 0);
                arrows[i] = arrow;
            }
        }
    }
}

    void CreateArrow(int index)
{
    // Don't create arrows for ball objects
    if (index >= 300 && index <= 302)
    {
        return;
    }

    GameObject player = GameObject.Find("Player" + index);
    if (player != null && arrowPrefab != null)
    {
        GameObject arrow = Instantiate(arrowPrefab);
        arrow.transform.SetParent(player.transform, false);
        arrow.transform.localPosition = new Vector3(0, -0.5f, 0);
        arrow.transform.localRotation = Quaternion.Euler(90, 0, 0);
        arrows[index] = arrow;
        arrow.SetActive(showPlayerArrows); // Set initial visibility
    }
    Debug.Log("Arrow created for player " + index);
}

void AdjustPlayer(int index, int leagueId)
    {
        if (playerHeights.ContainsKey(leagueId) && playerWeights.ContainsKey(leagueId))
        {
            GameObject player = GameObject.Find("Player" + index);
            if (player != null)
            {
                float height = playerHeights[leagueId];
                float weight = playerWeights[leagueId];
                player.transform.localScale = new Vector3(weight, height, weight);

                if (playerNumbers.ContainsKey(leagueId) && playerPositions.ContainsKey(leagueId) && textPrefab != null)
                {
                    GameObject textObj = Instantiate(textPrefab);
                    textObj.transform.SetParent(player.transform, false);
                    
                    // Adjust text position
                    float textHeight = height + 1.0f; // Increase this value to move the text higher
                    textObj.transform.localPosition = new Vector3(-1.45f, -0.24f, -0.16f);
                    
                    TextMeshPro tmpComponent = textObj.GetComponent<TextMeshPro>();
                    if (tmpComponent != null)
                    {
                        UpdatePlayerText(tmpComponent, leagueId);
                        
                        // Center-align the text
                        tmpComponent.alignment = TextAlignmentOptions.Center;
                        
                        // Adjust the size and spacing
                        tmpComponent.fontSize = 4; // Adjust this value as needed
                        tmpComponent.lineSpacing = -26; // Negative value to bring lines closer together
                        
                        // Enable text wrapping
                        tmpComponent.enableWordWrapping = true;
                        tmpComponent.overflowMode = TextOverflowModes.Overflow;
                    }
                    textObjects[index] = textObj;
                }
            }
        }
        //Debug.Log("Players adjusted.");
    }

    void UpdateAllPlayerTexts()
    {
        foreach (var kvp in textObjects)
        {
            TextMeshPro tmpComponent = kvp.Value.GetComponent<TextMeshPro>();
            if (tmpComponent != null)
            {
                int index = kvp.Key;
                int leagueId = GetLeagueIdForIndex(index);
                if (leagueId != -1)
                {
                    UpdatePlayerText(tmpComponent, leagueId);
                    
                    // Update text position
                    GameObject player = GameObject.Find("Player" + index);
                    if (player != null && playerHeights.ContainsKey(leagueId))
                    {
                        float height = playerHeights[leagueId];
                        float textHeight = height + 1.0f; // Increase this value to move the text higher
                        kvp.Value.transform.localPosition = new Vector3(-1.45f, -0.24f, -0.16f);
                    }
                }
                else
                {
                    //Debug.LogWarning($"Unable to update text for player with index {index}");
                }
            }
        }
    }

public void CyclePlayerInfoDisplayMode()
    {
        playerInfoDisplayMode = (PlayerInfoDisplayMode)(((int)playerInfoDisplayMode + 1) % 3);
        UpdateAllPlayerTexts();
        Debug.Log($"Player info display mode changed to: {playerInfoDisplayMode}");
    }

void UpdatePlayerText(TextMeshPro tmpComponent, int leagueId)
    {
        switch (playerInfoDisplayMode)
        {
            case PlayerInfoDisplayMode.NumberOnly:
                tmpComponent.text = $"{playerNumbers[leagueId]}";
                break;
            case PlayerInfoDisplayMode.NumberAndPosition:
                tmpComponent.text = $"{playerNumbers[leagueId]}\n<size=80%>({playerPositions[leagueId]})</size>";
                break;
            case PlayerInfoDisplayMode.FullInfo:
                tmpComponent.text = $"{playerNumbers[leagueId]}\n<size=80%>{leagueId} ({playerPositions[leagueId]})</size>";
                break;
        }
    }

    public void ChangePlayerInfoDisplayMode(PlayerInfoDisplayMode newMode)
    {
        playerInfoDisplayMode = newMode;
        UpdateAllPlayerTexts();
    }


    private Dictionary<int, int> indexToLeagueIdMap;

void CreateIndexToLeagueIdMap()
{
    indexToLeagueIdMap = new Dictionary<int, int>();
    if (myUniqueIdsList != null && myUniqueIdsList.league_teams != null)
    {
        foreach (var uniqueId in myUniqueIdsList.league_teams)
        {
            indexToLeagueIdMap[uniqueId.index] = uniqueId.league_id;
        }
    }
}

    int GetLeagueIdForIndex(int index)
{
    if (indexToLeagueIdMap.TryGetValue(index, out int leagueId))
    {
        return leagueId;
    }
    
    Debug.LogWarning($"No league ID found for index {index}");
    return -1;
}
            
   void Update()
{ 
    if (!gameStarted) 
        return;

    UpdatePlayerVelocityTexts();

    // Toggle velocity text visibility with 'V' key
    if (Input.GetKeyDown(KeyCode.V))
    {
        ToggleVelocityTextVisibility();
    }

    // Check for threshold switch key press
    if (Input.GetKeyDown(KeyCode.Tab)) // You can change KeyCode.Tab to any key you prefer
    {
        SwitchThresholdType();
    }

    // Add this to allow changing display mode with a key press
    if (Input.GetKeyDown(KeyCode.T))
    {
        Debug.Log("T key pressed");
        CyclePlayerInfoDisplayMode();
    }

    // Toggle ball arrow visibility with 'B' key
    if (Input.GetKeyDown(KeyCode.B))
    {
        showBallArrow = !showBallArrow;
        UpdateBallArrowVisibility();
        Debug.Log($"Ball arrow visibility toggled: {showBallArrow}");
    }

    // Toggle player arrow visibility with 'P' key
    if (Input.GetKeyDown(KeyCode.P))
    {
        showPlayerArrows = !showPlayerArrows;
        UpdatePlayerArrowVisibility();
        Debug.Log($"Player arrow visibility toggled: {showPlayerArrows}");
    }

    if (chooseEvent != lastChooseEvent)
    {
        ResetChosenEvent();
        lastChooseEvent = chooseEvent;
    }

    // Check if showBallArrow has changed
    if (showBallArrow != lastShowBallArrow)
    {
        UpdateBallArrowVisibility();
        lastShowBallArrow = showBallArrow;
    }

    // Check if showPlayerArrows has changed
    if (showPlayerArrows != lastShowPlayerArrows)
    {
        UpdatePlayerArrowVisibility();
        lastShowPlayerArrows = showPlayerArrows;
    }

    // Increment the elapsed time by the time passed since the last frame
    elapsedTime += Time.deltaTime;

    // Update game elapsed time based on update interval
    gameElapsedTime += Time.deltaTime / gameSpeed;
    
    gameTimer += Time.deltaTime;

    // Update the displayed timer text
    if (gametimerText != null)
    {
        gametimerText.text = $"Time: {Mathf.Floor(gameTimer)}s"; // Display game time in seconds
    }

    if (playAllEvents == 1)
    {
        // Display match time using event positional data
        /*if (currentTimeMsIndex < myEventPositionalDataList.event_data.Length)
        {
            var currentEventData = myEventPositionalDataList.event_data[currentTimeMsIndex];
            float match_time_minutes = currentEventData.match_time_seconds_column / 60f;
            // Display the match time in seconds
            timerText.text = $"Match Time: {match_time_minutes.ToString("F2")} minutes \n {currentEventData.match_time_seconds_column} seconds";
        }*/
    }
    else if (playMatch == 1)
    {
        // Display match time using reduced data
        if (currentTimeMsIndex < groupedData.Keys.Count)
        {
            int currentTimeMs = groupedData.Keys.ElementAt(currentTimeMsIndex);
            List<ReducedData> currentEpochData = groupedData[currentTimeMs];

            if (currentEpochData.Count > 0)
            {
                // Assuming all data points in the current epoch have the same match time
                int matchTimeSeconds = currentEpochData[0].match_time_seconds_column;
                float matchTimeMinutes = currentEpochData[0].match_time_minutes_column;

                // Display the match time in minutes and seconds
                timerText.text = $"Match Time: {matchTimeMinutes:F2} minutes \n{matchTimeSeconds} seconds";
            }
        }
    }

    // Check if enough time has passed to update the data
    if (elapsedTime >= gameSpeed)
    {
        elapsedTime = 0f; // Reset the elapsed time
        Debug.Log($"playAllEvents: {playAllEvents}, chooseEvent: {chooseEvent}, playMatch: {playMatch}");

        if (chooseEvent == 1)
        {
            Debug.Log("Attempting to move players based on chosen event");
            MovePlayersBasedOnChosenEvent();
        }
        else if (playAllEvents == 1)
        { 
            Debug.Log("Attempting to move players based on event positional data");
            MovePlayersBasedOnEventPositionalData();
        }
        else if (playMatch == 1)
        {
            Debug.Log("Attempting to move players based on grouped data");
            MovePlayersBasedOnGroupedData();
        }
        else
        {
            Debug.Log("No valid play mode selected");
        }

        // Call this method to check and update player count
        CheckPlayerCountOnField();
    }

    // Interpolate positions of all players
    foreach (var index in targetPositions.Keys)
    {
        if (lerpTimes[index] < 1f)
        {
            lerpTimes[index] += Time.deltaTime / gameSpeed;
            string playerName = "Player" + index;
            GameObject player = GameObject.Find(playerName);
            if (player != null)
            {
                //player.transform.position = Vector3.Lerp(startPositions[index], targetPositions[index], lerpTimes[index]);
                Vector3 newPosition = Vector3.Lerp(startPositions[index], targetPositions[index], lerpTimes[index]);
                player.transform.position = newPosition;
                //Debug.Log($"{playerName} interpolated position: {newPosition}");

             

                // Update rotation after position update
                RotatePlayerTowardsBall(index);
            }
        }
    }
}



private void UpdatePlayer(int matchingIndex, Vector3 newPosition, float v, float log_v, float sqrt_v, int throwTrigger)
{
    string playerName = "Player" + matchingIndex;
    GameObject player = GameObject.Find(playerName);

    if (player == null)
    {
        Debug.LogError($"Player object not found: {playerName}");
        return;
    }

    // Set up interpolation
    startPositions[matchingIndex] = player.transform.position;
    targetPositions[matchingIndex] = newPosition;
    lerpTimes[matchingIndex] = 0f;

    bool isBall = matchingIndex >= 300 && matchingIndex <= 302;
    bool isAtOrigin = newPosition.x == 0 && newPosition.y == 0 && newPosition.z == 0;

    // Update velocity text visibility
    PlayerVelocityText pvt = playerVelocityTexts.Find(p => p.playerIndex == matchingIndex);
    if (pvt != null)
    {
        pvt.velocityText.gameObject.SetActive(!isAtOrigin && showVelocityText);
    }

    // Update visibility
    Renderer renderer = player.GetComponent<Renderer>();
    if (renderer != null)
    {
        renderer.enabled = !isAtOrigin;
    }

    if (arrows.ContainsKey(matchingIndex))
    {
        arrows[matchingIndex].SetActive(!isAtOrigin && (isBall ? showBallArrow : showPlayerArrows));
    }
    if (textObjects.ContainsKey(matchingIndex))
    {
        textObjects[matchingIndex].SetActive(!isAtOrigin);
    }

    if (isAtOrigin)
    {
        return;
    }

  

    if (!isBall)
    {
        RotatePlayerTowardsBall(matchingIndex);

        Animator playerAnimator = player.GetComponent<Animator>();
        if (playerAnimator != null)
        {
            float speed;
            string stateDescription;

            float selectedV = GetSelectedVelocity(v, log_v, sqrt_v);
            ClusterCenters selectedThresholds = GetSelectedThresholds();

            if (selectedThresholds != null)
            {
                if (selectedV < selectedThresholds.idle_threshold)
                {
                    speed = 0f;
                    stateDescription = "Idle";
                }
                else if (selectedV < selectedThresholds.walk_threshold)
                {
                    speed = Mathf.Lerp(0f, 0.5f, (selectedV - selectedThresholds.idle_threshold) / (selectedThresholds.walk_threshold - selectedThresholds.idle_threshold));
                    stateDescription = "Transitioning to Walk";
                }
                else
                {
                    speed = Mathf.Lerp(0.5f, 1f, (selectedV - selectedThresholds.walk_threshold) / (selectedThresholds.run_threshold - selectedThresholds.walk_threshold));
                    stateDescription = "Transitioning to Run";
                }
            }
            else
            {
                // Use default thresholds if cluster centers are not available
                if (v < 0.1f)
                {
                    speed = 0f;
                    stateDescription = "Idle";
                }
                else if (v < 0.5f)
                {
                    speed = Mathf.Lerp(0f, 0.5f, (v - 0.1f) / 0.4f);
                    stateDescription = "Transitioning to Walk";
                }
                else
                {
                    speed = Mathf.Lerp(0.5f, 1f, (v - 0.5f) / 0.5f);
                    stateDescription = "Transitioning to Run";
                }
            }

            playerAnimator.SetFloat("Speed", speed);
            //Debug.Log($"{playerName}: v={v:F3}, State: {stateDescription}, Speed: {speed:F2}");

            // Handle throw animation
            if (throwTrigger == 1)
            {
                TriggerThrowAnimation(matchingIndex);
            }
        }
        else
        {
            Debug.LogWarning($"No Animator component found on {playerName}");
        }
    }

    UpdateArrow(player, matchingIndex, isBall, newPosition);
    UpdateTextPosition(matchingIndex, isBall);
    AdjustTransparency(player, newPosition);

    lastMovedTimes[matchingIndex] = Time.time;
}

private void DestroyPlayerVelocityTexts()
{
    foreach (var pvt in playerVelocityTexts)
    {
        if (pvt.velocityText != null)
        {
            Destroy(pvt.velocityText.gameObject);
        }
    }
    playerVelocityTexts.Clear();
}

void OnDisable()
{
    DestroyPlayerVelocityTexts();
}

private float GetSelectedVelocity(float v, float log_v, float sqrt_v)
{
    switch (currentThresholdType)
    {
        case ThresholdType.V:
            return v;
        case ThresholdType.LogV:
            return log_v;
        case ThresholdType.SqrtV:
            return sqrt_v;
        default:
            return v;
    }
}

private ClusterCenters GetSelectedThresholds()
{
    if (clusterCenters == null)
    {
        Debug.LogError("clusterCenters is null");
        return null;
    }

    ClusterCenters selected = null;
    switch (currentThresholdType)
    {
        case ThresholdType.V:
            selected = clusterCenters.cluster_centers_v;
            break;
        case ThresholdType.LogV:
            selected = clusterCenters.cluster_centers_log_v;
            break;
        case ThresholdType.SqrtV:
            selected = clusterCenters.cluster_centers_sqrt_v;
            break;
    }

    if (selected == null)
    {
        Debug.LogError($"Selected ClusterCenters for {currentThresholdType} is null");
    }
    else
    {
        Debug.Log($"Selected thresholds for {currentThresholdType}: Idle={selected.idle_threshold}, Walk={selected.walk_threshold}, Run={selected.run_threshold}");
    }

    return selected;
}

private void UpdatePlayerAnimations()
{
    foreach (var kvp in targetPositions)
    {
        int index = kvp.Key;
        if (index < 300 || index > 302) // Not a ball
        {
            GameObject player = GameObject.Find("Player" + index);
            if (player != null)
            {
                Animator animator = player.GetComponent<Animator>();
                if (animator != null)
                {
                    float v = 0f; // You need to get the current velocity for this player
                    float speed = CalculateSpeed(v);
                    animator.SetFloat("Speed", speed);
                }
            }
        }
    }
}

private float CalculateSpeed(float v)
{
    ClusterCenters selectedThresholds = GetSelectedThresholds();
    if (selectedThresholds == null) return 0f;

    if (v < selectedThresholds.idle_threshold)
    {
        return 0f;
    }
    else if (v < selectedThresholds.walk_threshold)
    {
        return Mathf.Lerp(0f, 0.5f, (v - selectedThresholds.idle_threshold) / (selectedThresholds.walk_threshold - selectedThresholds.idle_threshold));
    }
    else
    {
        return Mathf.Lerp(0.5f, 1f, (v - selectedThresholds.walk_threshold) / (selectedThresholds.run_threshold - selectedThresholds.walk_threshold));
    }
}

    private void TriggerThrowAnimation(int playerIndex)
    {
        GameObject player = GameObject.Find("Player" + playerIndex);
        if (player != null)
        {
            Animator animator = player.GetComponent<Animator>();
            if (animator != null)
            {
                //Debug.Log($"Setting Throw trigger for Player {playerIndex}");
                animator.SetTrigger("Throw");
            }
            else
            {
                //Debug.LogWarning($"No Animator component found on Player {playerIndex}");
            }
        }
        else
        {
            Debug.LogWarning($"Player object not found: Player {playerIndex}");
        }
    }

private void UpdateArrow(GameObject player, int matchingIndex, bool isBall, Vector3 newPosition)
{
    if (isBall)
    {
        if (showBallArrow)
        {
            if (!arrows.ContainsKey(matchingIndex))
            {
                GameObject arrow = Instantiate(arrowPrefab);
                arrow.transform.SetParent(player.transform, false);
                arrows[matchingIndex] = arrow;
            }
            GameObject ballArrow = arrows[matchingIndex];
            ballArrow.transform.localPosition = new Vector3(0, -0.25f, 0);
            Vector3 direction = (newPosition - player.transform.position).normalized;
            ballArrow.transform.rotation = Quaternion.LookRotation(direction) * Quaternion.Euler(90, -90, 0);
            ballArrow.SetActive(true);
        }
        else if (arrows.ContainsKey(matchingIndex))
        {
            arrows[matchingIndex].SetActive(false);
        }
    }
    else if (arrows.ContainsKey(matchingIndex))
    {
        GameObject arrow = arrows[matchingIndex];
        if (showPlayerArrows)
        {
            Vector3 direction = (newPosition - player.transform.position).normalized;
            arrow.transform.localPosition = Vector3.zero;
            arrow.transform.rotation = Quaternion.LookRotation(direction) * Quaternion.Euler(90, -90, 0);
            arrow.SetActive(true);
        }
        else
        {
            arrow.SetActive(false);
        }
    }
}

private void UpdateTextPosition(int matchingIndex, bool isBall)
{
    if (!isBall && textObjects.ContainsKey(matchingIndex))
    {
        GameObject textObj = textObjects[matchingIndex];
        textObj.transform.localPosition = new Vector3(-1.45f, -0.24f, -0.16f);
    }
}

private void AdjustTransparency(GameObject player, Vector3 newPosition)
{
    ChangeTransparency(player, IsOutOfBounds(newPosition) ? 0.7f : 1f);
}

void MovePlayersBasedOnGroupedData()
{
    if (currentTimeMsIndex < groupedData.Keys.Count)
    {
        int currentTimeMs = groupedData.Keys.ElementAt(currentTimeMsIndex);
        List<ReducedData> currentEpochData = groupedData[currentTimeMs];

        foreach (var data in currentEpochData)
        {
            int matchingIndex = GetMatchingIndex(data.league_id);

            if (matchingIndex != -1)
            {
                Vector3 newPosition = new Vector3(data.x, data.z, data.y);
                UpdatePlayer(matchingIndex, newPosition, data.v, data.log_v, data.sqrt_v, data.throw_trigger);
            }
        }

        currentTimeMsIndex++;
    }
}
   
    void MovePlayersBasedOnEventPositionalData()
{Debug.Log($"Starting MovePlayersBasedOnEventPositionalData. Current event index: {currentEventKeyIndex}, Total events: {eventKeys.Count}");
    if (currentEventKeyIndex < eventKeys.Count)
    {
        string currentEventKey = eventKeys[currentEventKeyIndex];
        Debug.Log($"Processing event key: {currentEventKey}");

        // Update the current event key text
        if (currentEventKeyText != null)
        {
            currentEventKeyText.text = $"Current Event Key: {currentEventKey}";
        }

        var timeData = eventData[currentEventKey];

        if (timeKeys == null || timeKeys.Count == 0)
        {
            timeKeys = new List<string>(timeData.Keys);
            timeKeys.Sort((a, b) => int.Parse(a).CompareTo(int.Parse(b)));
            currentTimeKeyIndex = 0;
        }

        if (currentTimeKeyIndex < timeKeys.Count)
        {
            string currentTimeKey = timeKeys[currentTimeKeyIndex];
            List<EventPositionalData> positionEvents = timeData[currentTimeKey];

            if (positionEvents.Count > 0)
            {
                // Use the first event's time for all events in this time step
                int matchTimeSeconds = int.Parse(currentTimeKey);
                float matchTimeMinutes = matchTimeSeconds / 60f;

                // Update timer display
                if (timerText != null)
                {
                    timerText.text = $"Match Time: {matchTimeMinutes:F2} minutes \n{matchTimeSeconds} seconds";
                }

                foreach (var playerEvent in positionEvents)
                    {
                        int matchingIndex = GetMatchingIndex(playerEvent.league_id);
                        if (matchingIndex != -1)
                        {
                            Vector3 newPosition = new Vector3(playerEvent.x, playerEvent.z, playerEvent.y);
                            UpdatePlayer(matchingIndex, newPosition, playerEvent.v, playerEvent.log_v, playerEvent.sqrt_v, playerEvent.throw_trigger);
                        }
                    }
            }

            currentTimeKeyIndex++;
        }
        else
        {
            currentEventKeyIndex++;
            timeKeys = null;
        }
    }
    else
    {
        // All events processed
        Debug.Log("All events completed");
        if (currentEventKeyText != null)
        {
            currentEventKeyText.text = "All Events Completed";
        }
    }
}

   void MovePlayersBasedOnChosenEvent()
{
    Debug.Log($"MovePlayersBasedOnChosenEvent called for event {chooseEvent}");

    // Update the current event key text
    if (currentEventKeyText != null)
    {
        currentEventKeyText.text = $"Current Event Key: {chooseEvent}";
    }

    Dictionary<string, Dictionary<string, List<EventPositionalData>>> eventDataToUse;

    if (eventKeyData.ContainsKey(chooseEvent))
    {
        eventDataToUse = eventKeyData[chooseEvent].event_data;
        Debug.Log($"Using individual event data for event {chooseEvent}");
    }
    else if (eventData != null && eventData.ContainsKey(chooseEvent.ToString()))
    {
        eventDataToUse = new Dictionary<string, Dictionary<string, List<EventPositionalData>>>
        {
            { chooseEvent.ToString(), eventData[chooseEvent.ToString()] }
        };
        Debug.Log($"Using grouped event data for event {chooseEvent}");
    }
    else
    {
        Debug.LogWarning($"No data found for event {chooseEvent}");
        if (currentEventKeyText != null)
        {
            currentEventKeyText.text = $"No data for Event: {chooseEvent}";
        }
        return;
    }

    if (eventDataToUse == null || eventDataToUse.Count == 0)
    {
        Debug.LogWarning($"Event data is empty for event {chooseEvent}");
        if (currentEventKeyText != null)
        {
            currentEventKeyText.text = $"Empty data for Event: {chooseEvent}";
        }
        return;
    }

    if (currentEventKeyIndex >= eventDataToUse.Count)
    {
        Debug.Log($"All data for chosen event {chooseEvent} has been processed");
        if (currentEventKeyText != null)
        {
            currentEventKeyText.text = $"Event {chooseEvent} Completed";
        }
        ResetChosenEvent();
        return;
    }

    string currentEventKey = eventDataToUse.Keys.ElementAt(currentEventKeyIndex);
    var timeData = eventDataToUse[currentEventKey];

    if (timeKeys == null || timeKeys.Count == 0)
    {
        timeKeys = new List<string>(timeData.Keys);
        timeKeys.Sort((a, b) => int.Parse(a).CompareTo(int.Parse(b)));
        currentTimeKeyIndex = 0;
    }

    if (currentTimeKeyIndex < timeKeys.Count)
    {
        string currentTimeKey = timeKeys[currentTimeKeyIndex];
        List<EventPositionalData> positionEvents = timeData[currentTimeKey];

        Debug.Log($"Processing Event: {chooseEvent}, Time: {currentTimeKey} seconds, Position events count: {positionEvents.Count}");

        foreach (var playerEvent in positionEvents)
    {
        int matchingIndex = GetMatchingIndex(playerEvent.league_id);
        if (matchingIndex != -1)
        {
            Vector3 newPosition = new Vector3(playerEvent.x, playerEvent.z, playerEvent.y);
            UpdatePlayer(matchingIndex, newPosition, playerEvent.v, playerEvent.log_v, playerEvent.sqrt_v, playerEvent.throw_trigger);
        }
        else
        {
            Debug.LogWarning($"No matching index found for league_id {playerEvent.league_id}");
        }
    }

        // Update timer display
        int matchTimeSeconds = int.Parse(currentTimeKey);
        float matchTimeMinutes = matchTimeSeconds / 60f;
        if (timerText != null)
        {
            timerText.text = $"Match Time: {matchTimeMinutes:F2} minutes \n{matchTimeSeconds} seconds";
        }

        currentTimeKeyIndex++;
    }
    else
    {
        currentEventKeyIndex++;
        timeKeys = null;
        if (currentEventKeyIndex >= eventDataToUse.Count)
        {
            Debug.Log($"All data for chosen event {chooseEvent} has been processed");
            if (currentEventKeyText != null)
            {
                currentEventKeyText.text = $"Event {chooseEvent} Completed";
            }
            ResetChosenEvent();
        }
    }
}

    bool IsOutOfBounds(Vector3 position)
    {
        return position.x < 0 || position.x > 40 || position.z < 0 || position.z > 20;
    }

    IEnumerator CheckInactivity()
    {
        while (true)
        {
            yield return new WaitForSeconds(1f); // Check every second

            float currentTime = Time.time;

            foreach (var kvp in lastMovedTimes)
            {
                int index = kvp.Key;
                float lastMovedTime = kvp.Value;

                // If player hasn't moved for more than 5 seconds
                if (currentTime - lastMovedTime > 5f)
                {
                    GameObject player = GameObject.Find("Player" + index);
                    if (player != null)
                    {
                        ChangeTransparency(player, 0.5f); // Make player transparent
                    }
                }
            }
        }
    }

    void ChangeTransparency(GameObject obj, float alpha) //Remember to change the Team1 and Team2 Material Rendering Mode to "Transparent" in Inspector in Unity
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer != null)
        {
            Color color = renderer.material.color;
            color.a = alpha;
            renderer.material.color = color;
        }
    }

int GetMatchingIndex(int leagueId)
{
    foreach (var uniqueId in myUniqueIdsList.league_teams)
    {
        if (uniqueId.league_id == leagueId)
        {
            return uniqueId.index;
        }
    }
    Debug.LogWarning($"No matching index found for league_id {leagueId}");
    return -1;
}
    void CreateBorder() // Create a border around the field
    {
        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.positionCount = 5; //The number of positions (vertices) the LineRenderer will use is set to 5. This includes the four corners of the rectangle and an additional point to close the loop.
        lineRenderer.startWidth = 0.1f;
        lineRenderer.endWidth = 0.1f;
        lineRenderer.loop = true;
        lineRenderer.useWorldSpace = true; //false: positions are relative to the parent game object's local space rather than the global world space.

        // Set the border positions
        Vector3[] positions = new Vector3[]
        {
            new Vector3(0, 0, 0),
            new Vector3(40, 0, 0),
            new Vector3(40, 0, 20),
            new Vector3(0, 0, 20),
            new Vector3(0, 0, 0) // Close the loop by returning to the start point
        };
        lineRenderer.SetPositions(positions);
    }

      // Check the number of players on the field for each team
    void CheckPlayerCountOnField()
    {
        team1PlayerCount = myUniqueIdsList.league_teams.Count(item => (targetPositions[item.index].x <= 40 && targetPositions[item.index].x != 0 && targetPositions[item.index].x > 0 && targetPositions[item.index].z <= 20 && targetPositions[item.index].z > 0 && targetPositions[item.index].z != 0) && item.team == 1);
        team2PlayerCount = myUniqueIdsList.league_teams.Count(item => (targetPositions[item.index].x <= 40 && targetPositions[item.index].x != 0 && targetPositions[item.index].x > 0 && targetPositions[item.index].z <= 20 && targetPositions[item.index].z > 0 && targetPositions[item.index].z != 0) && item.team == 2);

        //Debug.Log($"Team 1 Player Count: {team1PlayerCount}, Team 2 Player Count: {team2PlayerCount}");

        // Update the player count text and change its color if needed for Team 1
        team1PlayerCountText.text = "Team 1 Player Count on field: " + team1PlayerCount;
        if (team1PlayerCount > 7)
        {
            team1PlayerCountText.color = Color.red;
        }
        else
        {
            team1PlayerCountText.color = Color.black;
        }

        // Update the player count text and change its color if needed for Team 2
        team2PlayerCountText.text = "Team 2 Player Count on field: " + team2PlayerCount;
        if (team2PlayerCount > 7)
        {
            team2PlayerCountText.color = Color.red;
        }
        else
        {
            team2PlayerCountText.color = Color.black;
        }
    }
}

public static class GameObjectExtensions
{
    public static void SetData<T>(this GameObject gameObject, string key, T value)
    {
        var dataComponent = gameObject.GetComponent<ObjectData>() ?? gameObject.AddComponent<ObjectData>();
        dataComponent.SetData(key, value);
    }

    public static T GetData<T>(this GameObject gameObject, string key)
    {
        var dataComponent = gameObject.GetComponent<ObjectData>();
        return dataComponent != null ? dataComponent.GetData<T>(key) : default(T);
    }
}

public class ObjectData : MonoBehaviour
{
    private Dictionary<string, object> dataStore = new Dictionary<string, object>();

    public void SetData<T>(string key, T value)
    {
        dataStore[key] = value;
    }

    public T GetData<T>(string key)
    {
        if (dataStore.TryGetValue(key, out object value) && value is T typedValue)
        {
            return typedValue;
        }
        return default(T);
    }
}
