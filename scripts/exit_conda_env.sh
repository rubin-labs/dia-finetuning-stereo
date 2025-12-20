#!/bin/bash
# Exit conda environment when conda is deleted
# Use this when you've deleted miniforge3/conda but are still in a conda environment

echo "Cleaning up conda environment remnants..."

# Unset all conda environment variables
unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
unset CONDA_PYTHON_EXE
unset CONDA_SHLVL
unset CONDA_BASE
unset CONDA_EXE

# Remove conda paths from PATH
# Save original PATH if needed, but remove any miniforge3/conda paths
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniforge3 | grep -v conda | tr '\n' ':' | sed 's/:$//')

# Reset PS1 prompt to remove (dia311) prefix
# This will restore a basic prompt
if [ -n "$PS1" ]; then
    # Remove conda prompt modifier from PS1
    export PS1=$(echo "$PS1" | sed 's/([^)]*) //')
    # If PS1 is empty or still has issues, set a basic prompt
    if [ -z "$PS1" ] || [[ "$PS1" == *"dia311"* ]]; then
        export PS1='\u@\h:\w\$ '
    fi
fi

# Use system commands instead of conda environment commands
# Reset PATH to use system binaries first
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"

echo "✓ Conda environment cleaned up"
echo "Current PATH: $PATH"
echo "You may want to run: exec bash (or exec zsh) to start a fresh shell"


