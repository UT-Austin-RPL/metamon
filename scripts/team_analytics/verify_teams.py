#!/usr/bin/env python3
"""
Verify Team Files

Check exported team files for issues (too many Pokemon, etc.)
"""

import sys
from pathlib import Path
import argparse


def count_pokemon_in_file(filepath: Path) -> int:
    """
    Count Pokemon in a Showdown format team file.

    Returns:
        Number of Pokemon in the team
    """
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    pokemon_count = 0

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip empty lines, comments, moves, and abilities
        if not line:
            continue
        if line.startswith('#'):
            continue
        if line.startswith('-'):
            continue
        if line.startswith('Ability:'):
            continue
        if line.startswith('EVs:'):
            continue
        if line.startswith('IVs:'):
            continue
        if line.startswith('Level:'):
            continue
        if line.startswith('Shiny:'):
            continue
        if ':' in line and not line.startswith('Ability:'):
            # Nickname format "Nickname (Species)"
            continue

        # This looks like a Pokemon name
        # Make sure it's at the start of a Pokemon block
        if i == 0 or lines[i-1].strip() == '' or lines[i-1].strip().startswith('#'):
            pokemon_count += 1

    return pokemon_count


def verify_team_directory(team_dir: str, fix: bool = False) -> dict:
    """
    Verify all team files in a directory.

    Args:
        team_dir: Path to directory with team files
        fix: If True, attempt to fix problematic files

    Returns:
        Dictionary with verification results
    """
    team_path = Path(team_dir)

    if not team_path.exists():
        print(f"ERROR: Directory not found: {team_dir}")
        sys.exit(1)

    # Find all team files
    team_files = list(team_path.glob('*.gen*ou_team'))

    if not team_files:
        print(f"ERROR: No team files found in {team_dir}")
        sys.exit(1)

    print(f"Checking {len(team_files)} team files...\n")

    results = {
        'total': len(team_files),
        'valid': 0,
        'too_many': [],
        'too_few': [],
        'empty': []
    }

    for filepath in sorted(team_files):
        pokemon_count = count_pokemon_in_file(filepath)

        if pokemon_count == 0:
            results['empty'].append((filepath.name, pokemon_count))
            print(f"EMPTY: {filepath.name} (0 Pokemon)")
        elif pokemon_count > 6:
            results['too_many'].append((filepath.name, pokemon_count))
            print(f"ERROR: {filepath.name} has {pokemon_count} Pokemon (max 6)")

            if fix:
                fix_team_file(filepath)

        elif pokemon_count < 1:
            results['too_few'].append((filepath.name, pokemon_count))
            print(f"WARN: {filepath.name} has only {pokemon_count} Pokemon")
        else:
            results['valid'] += 1

    return results


def fix_team_file(filepath: Path):
    """
    Fix a team file with too many Pokemon by removing duplicates.
    """
    print(f"  → Attempting to fix {filepath.name}...")

    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Parse Pokemon blocks
    pokemon_blocks = []
    current_block = []
    seen_species = set()

    for line in lines:
        stripped = line.strip()

        # Start of new Pokemon (non-empty, non-comment, non-detail line at block boundary)
        if stripped and not stripped.startswith('#') and not stripped.startswith('-') and \
           not stripped.startswith('Ability:') and not stripped.startswith('EVs:') and \
           not stripped.startswith('IVs:') and not stripped.startswith('Level:'):

            # Check if this is start of new block
            if current_block and (not current_block[-1].strip() or current_block[-1].strip().startswith('#')):
                # Save previous block
                if current_block:
                    pokemon_blocks.append(current_block)
                current_block = [line]
            elif not current_block:
                # First Pokemon
                current_block = [line]
            else:
                # Continuation of current block (detail line)
                current_block.append(line)
        else:
            current_block.append(line)

    # Add last block
    if current_block:
        pokemon_blocks.append(current_block)

    # Rebuild with only first 6 unique Pokemon
    fixed_lines = []
    pokemon_count = 0

    # Add header comments
    for line in lines:
        if line.strip().startswith('#'):
            fixed_lines.append(line)
        else:
            break

    if fixed_lines:
        fixed_lines.append('')

    # Add Pokemon blocks (first 6 only, deduplicated)
    for block in pokemon_blocks:
        if pokemon_count >= 6:
            break

        # Get species name from first non-comment line
        species_name = None
        for line in block:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                species_name = stripped.lower()
                break

        if species_name and species_name not in seen_species:
            seen_species.add(species_name)
            fixed_lines.extend(block)
            pokemon_count += 1

    # Write fixed file
    with open(filepath, 'w') as f:
        f.write('\n'.join(fixed_lines))

    # Verify fix
    new_count = count_pokemon_in_file(filepath)
    print(f"  → Fixed! Now has {new_count} Pokemon")


def main():
    parser = argparse.ArgumentParser(
        description="Verify exported team files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check teams
  python scripts/team_analytics/verify_teams.py $METAMON_CACHE_DIR/teams/smogon_pass2_selected

  # Check and auto-fix
  python scripts/team_analytics/verify_teams.py $METAMON_CACHE_DIR/teams/smogon_pass2_selected --fix
        """
    )

    parser.add_argument(
        'team_dir',
        type=str,
        help='Directory containing team files'
    )

    parser.add_argument(
        '--fix',
        action='store_true',
        help='Automatically fix problematic files'
    )

    args = parser.parse_args()

    results = verify_team_directory(args.team_dir, fix=args.fix)

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total files:      {results['total']}")
    print(f"Valid files:      {results['valid']}")
    print(f"Too many Pokemon: {len(results['too_many'])}")
    print(f"Too few Pokemon:  {len(results['too_few'])}")
    print(f"Empty files:      {len(results['empty'])}")

    if results['too_many']:
        print(f"\nFiles with too many Pokemon:")
        for name, count in results['too_many'][:10]:
            print(f"  - {name} ({count} Pokemon)")
        if len(results['too_many']) > 10:
            print(f"  ... and {len(results['too_many']) - 10} more")

    if results['too_many'] and not args.fix:
        print("\nRun with --fix to automatically correct these files")
        sys.exit(1)
    elif results['valid'] == results['total']:
        print("\n✓ All team files are valid!")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
