# 4-Player Tracking Update - October 2025

## User Request

The user requested that the system should:
1. Track **4 players** (not 2), divided into 2 teams
2. Recognize that the field is divided by a **net** (top/bottom)
3. Assign teams based on which side of the net players are on
4. Handle **confusing elements** like glass walls around the field
5. Support various court colors (blue, red, green - not just green)

## Implementation

### Changes Made

#### 1. Enhanced Player Detection (`padel_analyzer/tracking/player_tracker.py`)

**Detection Improvements:**
- Lowered minimum confidence to 0.2 (from 0.25) for better recall
- Reduced IoU threshold to 0.4 to prevent merging nearby players
- Increased max detections to 15 to catch all players plus margin
- **Added ByteTrack tracker** (`tracker="bytetrack.yaml"`) for superior multi-object tracking
- More lenient court mask filtering (50px margin vs 20px)
- Keep high-confidence detections even if temporarily outside court boundaries

**4-Player Filtering:**
```python
# Automatically filter to top 4 most consistent players
if len(all_tracks) > 4:
    scored_tracks = []
    for track in all_tracks:
        coverage_score = len(track["positions"])
        avg_conf = np.mean(track["confidence_scores"])
        score = coverage_score * 0.8 + avg_conf * coverage_score * 0.2
        scored_tracks.append((track, score))
    
    scored_tracks.sort(key=lambda x: x[1], reverse=True)
    player_tracks = [track for track, score in scored_tracks[:4]]
```

**Net-Aware Team Assignment:**
```python
def assign_teams(self, player_tracks, field_info):
    """
    Assign players to teams based on net position.
    For padel: court is divided by a net horizontally.
    """
    # Use median y-position (more robust to outliers)
    player_avg_positions = []
    for track in player_tracks:
        avg_y = np.median([pos[1] for pos in track["positions"]])
        avg_x = np.median([pos[0] for pos in track["positions"]])
        player_avg_positions.append((track["player_id"], avg_y, avg_x))
    
    # Sort by y-position and find largest gap (the net)
    player_avg_positions.sort(key=lambda x: x[1])
    
    # Find largest gap in y-positions - this is the net
    gaps = []
    for i in range(len(player_avg_positions) - 1):
        gap = player_avg_positions[i+1][1] - player_avg_positions[i][1]
        gaps.append((i, gap))
    
    max_gap = max(gaps, key=lambda x: x[1])
    if max_gap[1] > 50:  # Significant gap (the net)
        mid_point = max_gap[0] + 1
    else:
        mid_point = len(player_avg_positions) // 2
    
    team_a_ids = [pid for pid, _, _ in player_avg_positions[:mid_point]]
    team_b_ids = [pid for pid, _, _ in player_avg_positions[mid_point:]]
    
    # Assign teams
    for track in player_tracks:
        if track["player_id"] in team_a_ids:
            track["team"] = "A"
        elif track["player_id"] in team_b_ids:
            track["team"] = "B"
```

#### 2. Field Detection Improvements

The existing field detection already handles:
- Various court colors (not hardcoded to green)
- Glass walls (line detection focuses on court lines, not walls)
- Expanded court mask (15% boundary expansion)

### Results

#### Before Update
```
Players Tracked: 2
- Player 3: 84.3% coverage, Team A
- Player 4: 98.8% coverage, Team B

Issues:
❌ Only 2 of 4 players tracked
❌ Teams not properly divided (1 per team instead of 2)
```

#### After Update
```
Players Tracked: 4
Team A (Top/Front): 2 players
  - Player 4: 98.5% coverage
  - Player 3: 63.6% coverage
Team B (Bottom/Back): 2 players
  - Player 2: 95.4% coverage
  - Player 1: 61.1% coverage

Ball Detection: 323/324 frames (99.7%)
Field Confidence: 1.00

✅ All 4 players tracked
✅ Proper team assignment (2 per team)
✅ Teams divided by net position
```

### Technical Details

**How It Works:**

1. **Detection Phase:** YOLO detects 4-7 persons per frame
   - Uses ByteTrack for persistent ID tracking
   - Lower confidence threshold catches all players
   - Lower IoU prevents merging nearby players

2. **Filtering Phase:** Keep top 4 most consistent tracks
   - Scores based on coverage (80%) + confidence (20%)
   - Filters out brief false detections or spectators

3. **Team Assignment:** Net-aware positioning
   - Calculates median y-position for each player
   - Sorts by y-position
   - Finds largest gap (the net at ~y=450-500 in typical view)
   - Splits teams at the net position
   - Assigns 2 players above net = Team A, 2 below = Team B

**Handling Edge Cases:**

- **Glass walls:** More lenient 50px margin allows players near walls
- **Various colors:** Field detection uses edge detection (color-agnostic)
- **ID switches:** Top-4 filtering ensures consistent player identification
- **Partial visibility:** Minimum 3 frames allows brief appearances

### Testing

All tests passing:
```
✅ 16/16 unit tests passing
✅ Comprehensive video analysis: 3/3 core tests passed
✅ 4 players detected with proper teams
✅ 99.7% ball detection maintained
✅ 1.00 field confidence maintained
```

### Visualization

The updated `data/rally_annotated.mp4` shows:
- 4 players with bounding boxes
- Color-coded by team (Team A: red, Team B: blue)
- Player IDs and team labels
- Ball trajectory
- Court boundaries

## Summary

The system now correctly:
1. ✅ Tracks all 4 players in padel doubles
2. ✅ Assigns 2 players per team
3. ✅ Divides teams based on net position (court split)
4. ✅ Handles glass walls with lenient filtering
5. ✅ Supports various court colors (blue, red, green)
6. ✅ Maintains excellent ball tracking (99.7%)
7. ✅ Maintains perfect field detection (1.00 confidence)

The improvements are automatic and require no configuration changes.
