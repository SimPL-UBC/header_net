"""
Unit tests for player-based spatial features extraction.
"""

import pytest
import numpy as np
from utils.player_features import (
    get_box_center,
    get_player_head,
    euclidean_distance,
    compute_distances_to_players,
    count_players_within_radius,
    compute_player_features,
    estimate_player_velocity,
    extract_player_features_from_frame_data,
    PLAYER_FEATURE_NAMES,
)


class TestBasicGeometry:
    """Test basic geometry functions."""

    def test_get_box_center(self):
        """Test box center calculation."""
        box = [100.0, 200.0, 50.0, 80.0]  # x, y, w, h
        cx, cy = get_box_center(box)
        assert cx == 125.0  # 100 + 50/2
        assert cy == 240.0  # 200 + 80/2

    def test_get_box_center_zero_size(self):
        """Test box center with zero dimensions."""
        box = [100.0, 200.0, 0.0, 0.0]
        cx, cy = get_box_center(box)
        assert cx == 100.0
        assert cy == 200.0

    def test_get_player_head(self):
        """Test head position (top center of bbox)."""
        box = [100.0, 200.0, 50.0, 80.0]
        hx, hy = get_player_head(box)
        assert hx == 125.0  # Center x
        assert hy == 200.0  # Top y (same as box y)

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        assert euclidean_distance(p1, p2) == 5.0  # 3-4-5 triangle

    def test_euclidean_distance_same_point(self):
        """Test distance between same point is zero."""
        p1 = (100.0, 200.0)
        assert euclidean_distance(p1, p1) == 0.0


class TestDistanceComputation:
    """Test distance computation functions."""

    def test_compute_distances_no_players(self):
        """Test with empty player list."""
        ball_center = (500.0, 300.0)
        players = []
        center_dists, head_dists, valid_players = compute_distances_to_players(
            ball_center, players
        )
        assert len(center_dists) == 0
        assert len(head_dists) == 0
        assert len(valid_players) == 0

    def test_compute_distances_single_player(self):
        """Test with one player."""
        ball_center = (500.0, 300.0)
        players = [
            {'box': [480.0, 260.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 1}
        ]
        center_dists, head_dists, valid_players = compute_distances_to_players(
            ball_center, players
        )

        assert len(center_dists) == 1
        assert len(head_dists) == 1
        assert len(valid_players) == 1

        # Player center: (480 + 20, 260 + 40) = (500, 300) - same as ball
        assert center_dists[0] == 0.0

        # Player head: (500, 260) - 40 pixels above ball
        assert head_dists[0] == 40.0

    def test_compute_distances_multiple_players(self):
        """Test with multiple players."""
        ball_center = (500.0, 300.0)
        players = [
            {'box': [400.0, 250.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 1},
            {'box': [550.0, 280.0, 40.0, 80.0], 'confidence': 0.8, 'class_id': 1},
        ]
        center_dists, head_dists, valid_players = compute_distances_to_players(
            ball_center, players
        )

        assert len(center_dists) == 2
        assert len(head_dists) == 2

    def test_count_players_within_radius(self):
        """Test counting players within radius."""
        distances = [30.0, 80.0, 150.0, 250.0]

        assert count_players_within_radius(distances, 50) == 1
        assert count_players_within_radius(distances, 100) == 2
        assert count_players_within_radius(distances, 200) == 3
        assert count_players_within_radius(distances, 300) == 4

    def test_count_players_within_radius_empty(self):
        """Test with no players."""
        assert count_players_within_radius([], 100) == 0


class TestPlayerFeatures:
    """Test main player feature computation."""

    def test_compute_player_features_no_players(self):
        """Test features when no players detected."""
        ball_data = {'box': [500.0, 300.0, 20.0, 20.0]}
        players = []

        features = compute_player_features(ball_data, players)

        assert features['player_count'] == 0
        assert features['dist_to_nearest_player'] == 9999.0
        assert features['dist_to_nearest_head'] == 9999.0
        assert features['num_players_50px'] == 0

    def test_compute_player_features_no_ball(self):
        """Test features when ball data is missing."""
        ball_data = {}  # No box
        players = [{'box': [400.0, 250.0, 40.0, 80.0]}]

        features = compute_player_features(ball_data, players)

        assert features['player_count'] == 0  # No features computed

    def test_compute_player_features_single_player(self):
        """Test features with one nearby player."""
        ball_data = {'box': [500.0, 300.0, 20.0, 20.0], 'velocity': [10.0, -5.0]}
        players = [
            {'box': [480.0, 260.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 1}
        ]

        features = compute_player_features(ball_data, players)

        assert features['player_count'] == 1
        assert features['dist_to_nearest_player'] >= 0
        assert features['dist_to_nearest_head'] >= 0
        assert features['goalkeeper_nearby'] == 0.0

    def test_compute_player_features_goalkeeper_nearby(self):
        """Test goalkeeper detection."""
        ball_data = {'box': [500.0, 300.0, 20.0, 20.0]}
        players = [
            {'box': [480.0, 260.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 3}  # Goalkeeper
        ]

        features = compute_player_features(ball_data, players)

        assert features['goalkeeper_nearby'] == 1.0

    def test_compute_player_features_goalkeeper_far(self):
        """Test goalkeeper not detected when far away."""
        ball_data = {'box': [100.0, 100.0, 20.0, 20.0]}
        players = [
            {'box': [500.0, 500.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 3}  # Far goalkeeper
        ]

        features = compute_player_features(ball_data, players)

        # Goalkeeper is more than 200px away
        assert features['goalkeeper_nearby'] == 0.0

    def test_compute_player_features_ball_above_head(self):
        """Test ball above nearest head indicator."""
        # Ball above player head
        ball_data = {'box': [500.0, 200.0, 20.0, 20.0]}  # Ball y=210 (center)
        players = [
            {'box': [490.0, 250.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 1}  # Head y=250
        ]

        features = compute_player_features(ball_data, players)

        # Ball center (210) < head (250), so ball is above
        assert features['ball_above_nearest_head'] == 1.0
        assert features['nearest_head_y_offset'] < 0  # Negative = ball above

    def test_compute_player_features_ball_below_head(self):
        """Test ball below nearest head."""
        ball_data = {'box': [500.0, 400.0, 20.0, 20.0]}  # Ball y=410 (center)
        players = [
            {'box': [490.0, 250.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 1}  # Head y=250
        ]

        features = compute_player_features(ball_data, players)

        # Ball center (410) > head (250), so ball is below
        assert features['ball_above_nearest_head'] == 0.0
        assert features['nearest_head_y_offset'] > 0  # Positive = ball below

    def test_compute_player_features_radius_counts(self):
        """Test player count at different radii."""
        ball_data = {'box': [500.0, 300.0, 20.0, 20.0]}
        players = [
            {'box': [500.0, 290.0, 20.0, 20.0], 'class_id': 1},  # Very close
            {'box': [430.0, 300.0, 20.0, 20.0], 'class_id': 1},  # ~70px away
            {'box': [350.0, 300.0, 20.0, 20.0], 'class_id': 1},  # ~150px away
        ]

        features = compute_player_features(ball_data, players)

        assert features['player_count'] == 3
        # Count depends on exact distances
        assert features['num_players_200px'] >= features['num_players_100px']
        assert features['num_players_100px'] >= features['num_players_50px']


class TestVelocityEstimation:
    """Test player velocity estimation."""

    def test_estimate_player_velocity_no_prev(self):
        """Test with no previous frame data."""
        current_player = {'box': [500.0, 300.0, 40.0, 80.0]}
        result = estimate_player_velocity(current_player, None)
        assert result is None

    def test_estimate_player_velocity_empty_prev(self):
        """Test with empty previous players list."""
        current_player = {'box': [500.0, 300.0, 40.0, 80.0]}
        result = estimate_player_velocity(current_player, [])
        assert result is None

    def test_estimate_player_velocity_match(self):
        """Test velocity estimation with matching player."""
        current_player = {'box': [510.0, 300.0, 40.0, 80.0]}  # Moved 10px right
        prev_players = [
            {'box': [500.0, 300.0, 40.0, 80.0]}  # Previous position
        ]

        vel = estimate_player_velocity(current_player, prev_players)

        assert vel is not None
        vx, vy = vel
        # 10px displacement at 25fps = 250 px/sec
        assert vx == 250.0
        assert vy == 0.0

    def test_estimate_player_velocity_no_match(self):
        """Test when player moves too far to match."""
        current_player = {'box': [500.0, 300.0, 40.0, 80.0]}
        prev_players = [
            {'box': [1000.0, 1000.0, 40.0, 80.0]}  # Too far to match
        ]

        vel = estimate_player_velocity(current_player, prev_players, max_match_distance=100.0)
        assert vel is None


class TestFrameDataExtraction:
    """Test extraction from frame metadata format."""

    def test_extract_player_features_from_frame_data(self):
        """Test extraction from metadata JSON frame entry format."""
        frame_data = {
            'offset': 0,
            'frame': 1000,
            'ball': {
                'box': [500.0, 300.0, 20.0, 20.0],
                'confidence': 0.9,
                'velocity': [5.0, -2.0]
            },
            'players': [
                {'box': [480.0, 260.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 1},
                {'box': [550.0, 280.0, 40.0, 80.0], 'confidence': 0.8, 'class_id': 1},
            ]
        }

        features = extract_player_features_from_frame_data(frame_data)

        assert features['player_count'] == 2
        assert features['dist_to_nearest_player'] < 9999.0
        assert features['dist_to_nearest_head'] < 9999.0

    def test_extract_player_features_from_frame_data_no_ball(self):
        """Test with missing ball data."""
        frame_data = {
            'offset': 0,
            'frame': 1000,
            'players': [
                {'box': [480.0, 260.0, 40.0, 80.0], 'confidence': 0.9, 'class_id': 1},
            ]
        }

        features = extract_player_features_from_frame_data(frame_data)

        # Should return default features
        assert features['player_count'] == 0


class TestFeatureNames:
    """Test feature name consistency."""

    def test_player_feature_names_count(self):
        """Test that we have expected number of player features."""
        assert len(PLAYER_FEATURE_NAMES) == 12

    def test_compute_returns_all_features(self):
        """Test that compute_player_features returns all named features."""
        ball_data = {'box': [500.0, 300.0, 20.0, 20.0]}
        players = [{'box': [480.0, 260.0, 40.0, 80.0], 'class_id': 1}]

        features = compute_player_features(ball_data, players)

        for name in PLAYER_FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"
