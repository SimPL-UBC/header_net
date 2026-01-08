"""
Utility functions for extracting player-based spatial features for header detection.

These features capture the spatial relationship between the ball and nearby players,
which is critical for identifying header events (ball must be near a player's head).
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


def get_box_center(box: List[float]) -> Tuple[float, float]:
    """
    Get center coordinates from a bounding box.

    Args:
        box: Bounding box as [x, y, w, h] where (x, y) is top-left corner

    Returns:
        Tuple of (center_x, center_y)
    """
    x, y, w, h = box[:4]
    return (x + w / 2, y + h / 2)


def get_player_head(box: List[float]) -> Tuple[float, float]:
    """
    Estimate head position from player bounding box.

    Assumes upright players with head at top center of bounding box.

    Args:
        box: Player bounding box as [x, y, w, h]

    Returns:
        Tuple of (head_x, head_y) - top center of bbox
    """
    x, y, w, h = box[:4]
    return (x + w / 2, y)  # Top center


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def compute_distances_to_players(
    ball_center: Tuple[float, float],
    players: List[Dict]
) -> Tuple[List[float], List[float], List[Dict]]:
    """
    Compute distances from ball to all players (center and head).

    Args:
        ball_center: Ball center coordinates (x, y)
        players: List of player detection dicts with 'box' key

    Returns:
        Tuple of:
        - List of distances to player centers
        - List of distances to player heads
        - List of player dicts (same order as distances)
    """
    center_distances = []
    head_distances = []
    valid_players = []

    for player in players:
        if 'box' not in player:
            continue

        box = player['box']
        player_center = get_box_center(box)
        player_head = get_player_head(box)

        center_distances.append(euclidean_distance(ball_center, player_center))
        head_distances.append(euclidean_distance(ball_center, player_head))
        valid_players.append(player)

    return center_distances, head_distances, valid_players


def count_players_within_radius(distances: List[float], radius: float) -> int:
    """
    Count players within a given radius.

    Args:
        distances: List of distances to players
        radius: Radius threshold in pixels

    Returns:
        Number of players within the radius
    """
    return sum(1 for d in distances if d <= radius)


def compute_player_features(
    ball_data: Dict,
    players: List[Dict],
    ball_velocity: Optional[Tuple[float, float]] = None,
    prev_players: Optional[List[Dict]] = None,
    radius_thresholds: List[int] = [50, 100, 200]
) -> Dict[str, float]:
    """
    Compute comprehensive player-based spatial features.

    Args:
        ball_data: Ball detection dict with 'box' and optionally 'velocity'
        players: List of player detection dicts
        ball_velocity: Ball velocity (vx, vy), overrides ball_data['velocity'] if provided
        prev_players: Previous frame's players for velocity estimation (optional)
        radius_thresholds: List of pixel thresholds for counting nearby players

    Returns:
        Dict of player-based features
    """
    features = {
        'dist_to_nearest_player': 9999.0,
        'dist_to_nearest_head': 9999.0,
        'num_players_50px': 0,
        'num_players_100px': 0,
        'num_players_200px': 0,
        'nearest_player_rel_vx': 0.0,
        'nearest_player_rel_vy': 0.0,
        'nearest_head_y_offset': 0.0,
        'ball_above_nearest_head': 0.0,
        'avg_player_density': 9999.0,
        'player_count': 0,
        'goalkeeper_nearby': 0.0,
    }

    if not players or 'box' not in ball_data:
        return features

    # Get ball position
    ball_center = get_box_center(ball_data['box'])
    ball_y = ball_center[1]

    # Get ball velocity
    if ball_velocity is not None:
        bvx, bvy = ball_velocity
    elif 'velocity' in ball_data and ball_data['velocity']:
        bvx, bvy = ball_data['velocity']
    else:
        bvx, bvy = 0.0, 0.0

    # Compute distances
    center_distances, head_distances, valid_players = compute_distances_to_players(
        ball_center, players
    )

    if not valid_players:
        return features

    # Basic counts
    features['player_count'] = len(valid_players)

    # Distance to nearest player (center)
    min_center_idx = np.argmin(center_distances)
    features['dist_to_nearest_player'] = center_distances[min_center_idx]

    # Distance to nearest head
    min_head_idx = np.argmin(head_distances)
    features['dist_to_nearest_head'] = head_distances[min_head_idx]

    # Nearest head vertical offset
    nearest_head_player = valid_players[min_head_idx]
    nearest_head = get_player_head(nearest_head_player['box'])
    features['nearest_head_y_offset'] = ball_y - nearest_head[1]  # Positive = ball below head
    features['ball_above_nearest_head'] = 1.0 if ball_y < nearest_head[1] else 0.0

    # Player counts within radius thresholds
    for threshold in radius_thresholds:
        count = count_players_within_radius(center_distances, threshold)
        features[f'num_players_{threshold}px'] = count

    # Average player density (mean distance to all players)
    features['avg_player_density'] = float(np.mean(center_distances))

    # Check for goalkeeper nearby (class_id == 3)
    for i, player in enumerate(valid_players):
        class_id = player.get('class_id', 1)
        if class_id == 3 and center_distances[i] <= 200:
            features['goalkeeper_nearby'] = 1.0
            break

    # Relative velocity of nearest player
    # For now, estimate as 0 unless we have tracking info
    # In future, could match players between frames for velocity estimation
    nearest_player = valid_players[min_center_idx]
    if prev_players:
        player_vel = estimate_player_velocity(nearest_player, prev_players)
        if player_vel:
            features['nearest_player_rel_vx'] = bvx - player_vel[0]
            features['nearest_player_rel_vy'] = bvy - player_vel[1]

    return features


def estimate_player_velocity(
    current_player: Dict,
    prev_players: List[Dict],
    max_match_distance: float = 100.0
) -> Optional[Tuple[float, float]]:
    """
    Estimate player velocity by matching to previous frame's detections.

    Uses simple nearest-neighbor matching based on position.

    Args:
        current_player: Current frame player detection
        prev_players: Previous frame player detections
        max_match_distance: Maximum distance for matching (pixels)

    Returns:
        Estimated velocity (vx, vy) or None if no match found
    """
    if not prev_players or 'box' not in current_player:
        return None

    curr_center = get_box_center(current_player['box'])

    # Find closest player in previous frame
    min_dist = float('inf')
    matched_prev = None

    for prev_player in prev_players:
        if 'box' not in prev_player:
            continue
        prev_center = get_box_center(prev_player['box'])
        dist = euclidean_distance(curr_center, prev_center)
        if dist < min_dist:
            min_dist = dist
            matched_prev = prev_player

    if matched_prev is None or min_dist > max_match_distance:
        return None

    prev_center = get_box_center(matched_prev['box'])

    # Velocity is displacement per frame (assume 1 frame = 1/25 sec)
    fps = 25.0
    vx = (curr_center[0] - prev_center[0]) * fps
    vy = (curr_center[1] - prev_center[1]) * fps

    return (vx, vy)


def extract_player_features_from_frame_data(
    frame_data: Dict,
    prev_frame_data: Optional[Dict] = None,
    radius_thresholds: List[int] = [50, 100, 200]
) -> Dict[str, float]:
    """
    Extract player features from a single frame's metadata.

    This is a convenience function for processing metadata JSON frame entries.

    Args:
        frame_data: Frame entry from metadata JSON with 'ball' and 'players' keys
        prev_frame_data: Previous frame data for velocity estimation
        radius_thresholds: Pixel thresholds for player counting

    Returns:
        Dict of player-based features
    """
    ball_data = frame_data.get('ball', {})
    players = frame_data.get('players', [])

    ball_velocity = None
    if 'velocity' in ball_data:
        ball_velocity = tuple(ball_data['velocity'])

    prev_players = None
    if prev_frame_data:
        prev_players = prev_frame_data.get('players', [])

    return compute_player_features(
        ball_data=ball_data,
        players=players,
        ball_velocity=ball_velocity,
        prev_players=prev_players,
        radius_thresholds=radius_thresholds
    )


# Feature names for the player-based features (in order)
PLAYER_FEATURE_NAMES = [
    'dist_to_nearest_player',
    'dist_to_nearest_head',
    'num_players_50px',
    'num_players_100px',
    'num_players_200px',
    'nearest_player_rel_vx',
    'nearest_player_rel_vy',
    'nearest_head_y_offset',
    'ball_above_nearest_head',
    'avg_player_density',
    'player_count',
    'goalkeeper_nearby',
]
