#!/usr/bin/env python3
"""
Test Export Functionality

Quick test to verify team export works correctly.
"""

import sys
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.team_analytics.export import TeamExporter


def test_numpy_serialization():
    """Test that numpy types are correctly serialized to JSON."""
    print("Testing numpy serialization...")

    # Create sample data with numpy types
    test_data = {
        'team_hash': 'test123',
        'species': np.array(['Pikachu', 'Charizard', 'Blastoise']),
        'stats': {
            'win_rate': np.float64(0.65),
            'total_battles': np.int64(100),
            'wins': np.int64(65)
        }
    }

    # Test conversion
    converted = TeamExporter._convert_to_serializable(test_data)

    # Try to JSON serialize
    try:
        json_str = json.dumps(converted, indent=2)
        print("✓ Numpy serialization works!")
        print(f"Sample output:\n{json_str}\n")
        return True
    except TypeError as e:
        print(f"✗ Numpy serialization failed: {e}")
        return False


def test_showdown_export():
    """Test Showdown format export."""
    print("Testing Showdown format export with duplicate team compositions...")

    # Create sample DataFrame with SAME team composition but DIFFERENT movesets
    # This simulates the real issue: same 6 Pokemon, different movesets
    teams_df = pd.DataFrame([
        {
            'team_hash': 'abc123',
            'team_species': np.array(['Alakazam', 'Tauros', 'Snorlax', 'Exeggutor', 'Chansey', 'Starmie']),
            'player_team_json': json.dumps([
                {'species': 'Alakazam', 'ability': 'noability', 'moves': ['psychic', 'recover', 'seismictoss', 'thunderwave']},
                {'species': 'Tauros', 'ability': 'noability', 'moves': ['bodyslam', 'hyperbeam', 'earthquake', 'blizzard']},
                {'species': 'Snorlax', 'ability': 'noability', 'moves': ['bodyslam', 'amnesia', 'rest', 'earthquake']},
                {'species': 'Exeggutor', 'ability': 'noability', 'moves': ['psychic', 'sleeppowder', 'explosion', 'stunspore']},
                {'species': 'Chansey', 'ability': 'noability', 'moves': ['softboiled', 'thunderwave', 'seismictoss', 'icebeam']},
                {'species': 'Starmie', 'ability': 'noability', 'moves': ['surf', 'thunderbolt', 'recover', 'blizzard']}
            ]),
            'win_rate': 0.68,
            'total_battles': 50,
            'wins': 34,
            'avg_turns': 42.5
        },
        {
            'team_hash': 'abc123',  # SAME hash (same species)
            'team_species': np.array(['Alakazam', 'Tauros', 'Snorlax', 'Exeggutor', 'Chansey', 'Starmie']),  # SAME species
            'player_team_json': json.dumps([
                {'species': 'Alakazam', 'ability': 'noability', 'moves': ['psychic', 'thunderwave', 'reflect', 'substitute']},  # DIFFERENT moves
                {'species': 'Tauros', 'ability': 'noability', 'moves': ['bodyslam', 'earthquake', 'hyperbeam', 'fireblast']},  # DIFFERENT moves
                {'species': 'Snorlax', 'ability': 'noability', 'moves': ['bodyslam', 'selfdestruct', 'rest', 'icebeam']},  # DIFFERENT moves
                {'species': 'Exeggutor', 'ability': 'noability', 'moves': ['psychic', 'megadrain', 'explosion', 'leechseed']},
                {'species': 'Chansey', 'ability': 'noability', 'moves': ['softboiled', 'toxic', 'seismictoss', 'thunderwave']},
                {'species': 'Starmie', 'ability': 'noability', 'moves': ['surf', 'thunderbolt', 'icebeam', 'recover']}
            ]),
            'win_rate': 0.72,
            'total_battles': 45,
            'wins': 32,
            'avg_turns': 38.2
        },
        {
            'team_hash': 'abc123',  # SAME hash again (3rd variant)
            'team_species': np.array(['Alakazam', 'Tauros', 'Snorlax', 'Exeggutor', 'Chansey', 'Starmie']),
            'player_team_json': json.dumps([
                {'species': 'Alakazam', 'ability': 'noability', 'moves': ['psychic', 'recover', 'thunderwave', 'reflect']},
                {'species': 'Tauros', 'ability': 'noability', 'moves': ['bodyslam', 'hyperbeam', 'blizzard', 'earthquake']},
                {'species': 'Snorlax', 'ability': 'noability', 'moves': ['bodyslam', 'rest', 'earthquake', 'selfdestruct']},
                {'species': 'Exeggutor', 'ability': 'noability', 'moves': ['psychic', 'sleeppowder', 'megadrain', 'stunspore']},
                {'species': 'Chansey', 'ability': 'noability', 'moves': ['softboiled', 'seismictoss', 'icebeam', 'thunderwave']},
                {'species': 'Starmie', 'ability': 'noability', 'moves': ['surf', 'recover', 'thunderbolt', 'blizzard']}
            ]),
            'win_rate': 0.65,
            'total_battles': 40,
            'wins': 26,
            'avg_turns': 44.1
        }
    ])

    # Test export
    temp_dir = tempfile.mkdtemp()
    try:
        files = TeamExporter.export_teams_to_showdown_format(
            teams_df,
            temp_dir,
            battle_format='gen1ou',
            include_stats=True
        )

        if files:
            expected_files = 3  # We have 3 teams with same composition but different movesets
            print(f"✓ Created {len(files)} team file(s) (expected {expected_files})")

            if len(files) != expected_files:
                print(f"✗ ERROR: Expected {expected_files} files but got {len(files)}")
                print(f"Files: {[Path(f).name for f in files]}")
                return False

            # Check that filenames are unique and follow naming pattern
            filenames = [Path(f).name for f in files]
            expected_pattern = ['team_abc123_Alakazam_Tauros_Snorlax.gen1ou_team',
                              'team_abc123_Alakazam_Tauros_Snorlax_v2.gen1ou_team',
                              'team_abc123_Alakazam_Tauros_Snorlax_v3.gen1ou_team']

            if sorted(filenames) == sorted(expected_pattern):
                print(f"✓ Filenames correctly numbered to avoid duplicates")
            else:
                print(f"✗ ERROR: Unexpected filenames")
                print(f"Expected (sorted): {sorted(expected_pattern)}")
                print(f"Got (sorted): {sorted(filenames)}")
                return False

            # Verify each file has different movesets
            movesets = []
            for file in files:
                with open(file, 'r') as f:
                    content = f.read()
                    # Extract moves (lines starting with -)
                    moves = [line.strip() for line in content.split('\n') if line.strip().startswith('-')]
                    movesets.append(set(moves))

            # Check that movesets are different
            if len(movesets) == 3:
                if movesets[0] != movesets[1] and movesets[1] != movesets[2] and movesets[0] != movesets[2]:
                    print(f"✓ Each team variant has different movesets")
                else:
                    print(f"✗ ERROR: Teams have identical movesets (should be different)")
                    return False

            # Read and display first file
            with open(files[0], 'r') as f:
                content = f.read()

            # Count Pokemon in the output
            pokemon_count = 0
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line and not line.startswith('#') and not line.startswith('-') and not line.startswith('Ability:'):
                    if i == 0 or lines[i-1] == '' or lines[i-1].startswith('#'):
                        pokemon_count += 1

            print(f"✓ Each team has {pokemon_count} Pokemon (should be ≤ 6)")
            if pokemon_count > 6:
                print(f"✗ ERROR: Team has {pokemon_count} Pokemon, exceeds limit of 6!")
                return False

            # Check that Gen 1 teams don't have Ability lines (abilities don't exist in Gen 1)
            if 'Ability:' in content:
                print(f"✗ ERROR: Gen 1 team contains 'Ability:' lines (abilities don't exist in Gen 1)")
                return False
            else:
                print(f"✓ No 'Ability:' lines (correct for Gen 1)")

            # Check that there are no comment lines (can confuse poke-env parser)
            if content.strip().startswith('#'):
                print(f"✗ ERROR: Team file starts with comment lines (can confuse parser)")
                return False
            else:
                print(f"✓ No comment header lines (clean format)")

            print(f"\nSample output (first variant):\n{content[:500]}...\n")
            return True
        else:
            print("✗ No files created")
            return False

    except Exception as e:
        print(f"✗ Showdown export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_export():
    """Test JSON export."""
    print("Testing JSON export...")

    # Create sample DataFrame
    teams_df = pd.DataFrame([
        {
            'team_hash': 'xyz789',
            'team_species': np.array(['Alakazam', 'Tauros', 'Snorlax']),
            'player_team_json': json.dumps([
                {'species': 'Alakazam', 'moves': ['psychic', 'recover']},
                {'species': 'Tauros', 'moves': ['bodyslam', 'hyperbeam']},
                {'species': 'Snorlax', 'moves': ['bodyslam', 'amnesia']}
            ]),
            'win_rate': 0.75,
            'total_battles': 20,
            'wins': 15,
            'avg_turns': 38.2
        }
    ])

    # Test export
    temp_dir = tempfile.mkdtemp()
    try:
        files = TeamExporter.export_teams_to_json(
            teams_df,
            temp_dir,
            filename_prefix='team'
        )

        if files:
            print(f"✓ Created {len(files)} JSON file(s)")
            # Read and display first file
            with open(files[0], 'r') as f:
                content = json.load(f)
            print(f"Sample output:\n{json.dumps(content, indent=2)}\n")
            return True
        else:
            print("✗ No files created")
            return False

    except Exception as e:
        print(f"✗ JSON export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 80)
    print("TEAM EXPORT FUNCTIONALITY TESTS")
    print("=" * 80 + "\n")

    results = []

    # Run tests
    results.append(("Numpy Serialization", test_numpy_serialization()))
    results.append(("Showdown Format Export", test_showdown_export()))
    results.append(("JSON Export", test_json_export()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed\n")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
